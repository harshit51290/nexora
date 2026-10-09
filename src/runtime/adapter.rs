//! Shared contract for all inference runtimes.
//!
//! Implements `docs/06-RUNTIME-ADAPTERS.md` §6.1 and the rules in `AGENTS.md`:
//!
//! * Rust = orchestration only. Every adapter drives its backend as a
//!   **supervised child process** (`std::process::Command`); no tensors,
//!   CUDA calls, or diffusion code ever live here.
//! * UI contract: callers use **only** `prepare` → `run` (plus `stop` /
//!   `health_check` for lifecycle). The UI never branches on backend kind
//!   except in Advanced mode.
//! * Each runtime owns an **isolated** interpreter resolved through
//!   [`crate::env::EnvManager`] (shared-reuse resolver with a
//!   convention-path fallback). Global `pip` is never touched.
//! * Every failure is a [`RuntimeError`] carrying a stable machine-readable
//!   `code` ([`codes`]) plus a human `hint` (`docs/11-UI-UX.md` §11.6 — no
//!   bare stack traces). [`RuntimeError`] converts into the core
//!   [`crate::core::NexoraError`] via `From` (see the mapping table there)
//!   so scheduler / API layers can use `?` against the core `Result`.
//! * Model metadata comes from the canonical
//!   [`crate::model::manager::ModelRecord`] (SQLite `models` row,
//!   `docs/04-DATA-MODEL.md`); the on-disk snapshot dir is passed
//!   separately as `model_dir` because the record carries no path.
//! * Request / result / pin types are `serde`-serializable; the job files
//!   handed to `serve.py` are produced with `serde_json`.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::core::NexoraError;
use crate::env::{EnvHandle, EnvManager};
use crate::model::manager::ModelRecord;

/// Stable, machine-readable error codes surfaced to the UI, logs, and API.
///
/// These codes are part of the persisted ERROR contract (`docs/04` §4.2):
/// never rename them. [`RuntimeError::hint`] maps each code to a human
/// cause + fix, and the `From<RuntimeError> for NexoraError` impl maps each
/// code into the core catalog (codes that have no core variant yet travel
/// inside the variant's free-text fields with the stable code embedded —
/// see NEED-CORE-ERRORS in that impl).
pub mod codes {
    /// Isolated env (`environments/<id>`) is missing or has no interpreter.
    pub const ENV_MISSING: &str = "E-ENV-MISSING";
    /// `bootstrap.py` failed while creating/upgrading the isolated env.
    pub const ENV_INSTALL_FAILED: &str = "E-ENV-INSTALL-FAILED";
    /// Runtime bundle (`runtimes/<id>/`) is not installed.
    pub const RUNTIME_NOT_INSTALLED: &str = "E-RUNTIME-NOT-INSTALLED";
    /// Env exists but the runtime is not ready to serve (e.g. serve
    /// entrypoint not bundled yet, weights not validated).
    pub const RUNTIME_NOT_READY: &str = "E-RUNTIME-NOT-READY";
    /// Supervised child process exited non-zero or was killed.
    pub const RUNTIME_CRASHED: &str = "E-RUNTIME-CRASHED";
    /// The child process could not be spawned at all.
    pub const CHILD_SPAWN_FAILED: &str = "E-CHILD-SPAWN-FAILED";
    /// Model directory is missing or unreadable.
    pub const MODEL_NOT_FOUND: &str = "E-MODEL-NOT-FOUND";
    /// Architecture / layout is not supported by this runtime.
    pub const MODEL_UNSUPPORTED: &str = "E-MODEL-UNSUPPORTED";
    /// Repo ships custom `.py` / `trust_remote_code` and the user has not
    /// consented (View / Sandbox / Cancel modal, `docs/09-ENV-SECURITY.md`).
    pub const CUSTOM_CODE_BLOCKED: &str = "E-CUSTOM-CODE-BLOCKED";
    /// Request must be delegated to the ComfyUI adapter (video workflows,
    /// complex graphs). Not a failure — a routing instruction. Callers must
    /// check [`RuntimeError::is_routing`] *before* converting into
    /// [`NexoraError`], or the routing signal is flattened into `E_OTHER`.
    pub const USE_COMFYUI: &str = "E-USE-COMFYUI";
    /// Estimated VRAM exceeds the active hardware profile (4 GB reference).
    pub const VRAM_LOW: &str = "E-VRAM-LOW";
}

/// A single runtime failure: machine code + technical message + human fix.
///
/// `scope` is the log scope from `docs/11-UI-UX.md` §11.6
/// (`Runtime` / `Download` / `Model` / `System`). `runtime` records which
/// adapter raised the error and feeds the [`NexoraError::RuntimeCrash`]
/// conversion.
#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub code: &'static str,
    pub message: String,
    pub hint: String,
    pub scope: &'static str,
    pub runtime: Option<String>,
}

impl RuntimeError {
    pub fn new(code: &'static str, message: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            hint: hint.into(),
            scope: "Runtime",
            runtime: None,
        }
    }

    /// Attach the raising adapter id (e.g. `"diffusers"`). Used for crash
    /// errors so the [`NexoraError`] conversion carries a real runtime name.
    pub fn with_runtime(mut self, id: &str) -> Self {
        self.runtime = Some(id.to_string());
        self
    }

    /// The stable wire code (`E-ENV-MISSING`, …). Never changes; safe to
    /// persist and to match on across versions.
    pub fn stable_code(&self) -> &'static str {
        self.code
    }

    /// True for [`codes::USE_COMFYUI`]: a re-dispatch instruction, not a
    /// failure. Check this before converting into [`NexoraError`].
    pub fn is_routing(&self) -> bool {
        self.code == codes::USE_COMFYUI
    }

    pub fn env_missing(env_id: &str, env_dir: &Path) -> Self {
        Self::new(
            codes::ENV_MISSING,
            format!(
                "isolated env '{env_id}' has no interpreter at {}",
                env_dir.display()
            ),
            "Open Environments, pick this runtime and press Install. \
             The app creates an isolated Python env — your system Python is never modified.",
        )
    }

    pub fn model_not_found(dir: &Path) -> Self {
        let mut e = Self::new(
            codes::MODEL_NOT_FOUND,
            format!("model directory not found: {}", dir.display()),
            "Re-download the model from its Model page, or check that the data \
             directory (Settings → Storage) still points at the right drive.",
        );
        e.scope = "Model";
        e
    }

    pub fn unsupported(what: impl Into<String>) -> Self {
        Self::new(
            codes::MODEL_UNSUPPORTED,
            what.into(),
            "This runtime cannot run the detected architecture. Check the model page \
             for the recommended runtime, or enable Experimental Mode (Advanced) to \
             try the Custom Python adapter with explicit consent.",
        )
    }

    pub fn crashed(detail: impl Into<String>, stderr_tail: &str) -> Self {
        let mut message = detail.into();
        if !stderr_tail.is_empty() {
            message.push_str(&format!(" — stderr: {stderr_tail}"));
        }
        // Error translation per docs/11-UI-UX.md §11.6.
        let hint = if looks_like_cuda_oom(stderr_tail) {
            "Not enough GPU memory. Try CPU offload, lower precision (FP16), a smaller \
             resolution / context length, or a quantized (Q4/Q5) variant of the model."
        } else {
            "The runtime process crashed but the app is still alive. Open Runtime → Logs \
             (View Technical Logs) for the full traceback, then retry or reinstall the runtime env."
        };
        Self::new(codes::RUNTIME_CRASHED, message, hint.to_string())
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} Fix: {}", self.code, self.message, self.hint)
    }
}

impl std::error::Error for RuntimeError {}

/// True when `haystack` (child stderr) looks like a CUDA OOM: "out of
/// memory" / "CUDA error" / CUDA allocation failure. Drives both the
/// human hint in [`RuntimeError::crashed`] and the [`NexoraError::CudaOom`]
/// mapping below.
pub fn looks_like_cuda_oom(haystack: &str) -> bool {
    let lower = haystack.to_lowercase();
    lower.contains("out of memory")
        || lower.contains("cuda error")
        || (lower.contains("cuda") && lower.contains("alloc"))
}

// NEED-CORE-ERRORS (integrator, `src/core/error.rs`, out of scope here):
// the core catalog has no EnvMissing / RuntimeNotInstalled / RuntimeNotReady
// / ModelNotFound / UseComfyUI-routing variants yet, so those codes travel
// inside `Other`'s message with the stable code embedded (`[E-...] … Fix:
// …`). When the variants land, narrow this mapping and drop the prefix.
// `CustomCode.repo` likewise has no structured source here, so the repo
// context travels inside `detail` (marked "see detail").
//
// Interim mapping (stable code + human fix are never dropped):
// | RuntimeError code                          | NexoraError variant |
// | E-CUSTOM-CODE-BLOCKED                      | CustomCode          |
// | E-RUNTIME-CRASHED / E-CHILD-SPAWN-FAILED   | CudaOom when
// |   (CUDA-OOM signature)                     |   looks_like_cuda_oom |
// | E-RUNTIME-CRASHED (other)                  | RuntimeCrash        |
// | everything else (incl. E-ENV-MISSING,      | Other (message keeps |
// |   E-USE-COMFYUI, E-VRAM-LOW)               | `[CODE] … Fix: …`)   |
impl From<RuntimeError> for NexoraError {
    fn from(e: RuntimeError) -> Self {
        let tagged = format!("[{}] {}", e.code, e.message);
        let tagged_fix = format!("[{}] {} Fix: {}", e.code, e.message, e.hint);
        match e.code {
            codes::CUSTOM_CODE_BLOCKED => NexoraError::CustomCode {
                repo: "(see detail)".to_string(),
                detail: tagged_fix,
            },
            codes::RUNTIME_CRASHED | codes::CHILD_SPAWN_FAILED
                if looks_like_cuda_oom(&e.message) =>
            {
                NexoraError::CudaOom { op: tagged }
            }
            codes::RUNTIME_CRASHED => NexoraError::RuntimeCrash {
                runtime: e.runtime.unwrap_or_else(|| "unknown".to_string()),
                pid: None,
                detail: tagged_fix,
            },
            _ => NexoraError::Other(anyhow::anyhow!("{}", tagged_fix)),
        }
    }
}

/// Convenience alias used by every adapter method.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

// ---------------------------------------------------------------------------
// Model metadata helpers.
//
// Adapters take the canonical [`ModelRecord`] plus the on-disk snapshot dir
// (`models/<id>`, resolved by the caller — NEED-STORAGE-PATH: there is no
// canonical snapshot-path helper in `src/storage` yet). Analyzer-fed fields
// that used to live on the old local `Model` struct (`architectures`,
// `pipeline_tag`, `library_name`) are re-derived from the snapshot files
// below; `capabilities` / `task` / `trust_level` come from the record.
// ---------------------------------------------------------------------------

/// True when `model_dir/<name>` exists (e.g. `config.json`,
/// `model_index.json`, `custom_model.py`).
pub fn has_file(model_dir: &Path, name: &str) -> bool {
    model_dir.join(name).is_file()
}

/// True when any file directly under `model_dir` (or one level of
/// subdirectories, covering HF `blobs/` layouts) ends with `ext`.
pub fn has_extension(model_dir: &Path, ext: &str) -> bool {
    has_extension_in(model_dir, ext, 2)
}

fn has_extension_in(dir: &Path, ext: &str, depth: u32) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if (path.is_file()
            && path.extension().and_then(|e| e.to_str()) == Some(ext.trim_start_matches('.')))
            || (path.is_dir() && depth > 0 && has_extension_in(&path, ext, depth - 1))
        {
            return true;
        }
    }
    false
}

/// Parse a small metadata file next to the weights (`config.json`,
/// `model_index.json`) into a JSON value. `None` when the file is missing
/// or unparseable — adapters treat that as "no evidence", never fatal.
pub fn read_meta_json(model_dir: &Path, name: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(model_dir.join(name)).ok()?;
    serde_json::from_str(&text).ok()
}

/// `config.json → architectures` (e.g. `["LlamaForCausalLM"]`). Empty when
/// the file is missing or has no string array under that key.
pub fn config_architectures(model_dir: &Path) -> Vec<String> {
    match read_meta_json(model_dir, "config.json") {
        Some(v) => v
            .get("architectures")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        None => Vec::new(),
    }
}

/// `model_index.json → _class_name` (e.g. `"StableDiffusionXLPipeline"`).
pub fn pipeline_class_name(model_dir: &Path) -> Option<String> {
    read_meta_json(model_dir, "model_index.json").and_then(|v| {
        v.get("_class_name")
            .and_then(|c| c.as_str())
            .map(str::to_string)
    })
}

// ---------------------------------------------------------------------------
// Inference types (UI builds these; adapters consume them).
// ---------------------------------------------------------------------------

/// One generation request. The UI builds this from the capability-driven
/// form (`docs/11-UI-UX.md` §11.4) and calls `prepare` → `run`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Canonical model row (`models` table).
    pub model: ModelRecord,
    /// On-disk snapshot dir (`models/<id>`). The record carries no path, so
    /// the caller resolves it (via `StorageLayout::models()`) and passes it.
    pub model_dir: PathBuf,
    /// Main prompt / input text.
    pub prompt: String,
    /// Diffusers-style negative prompt, if the capability form provides one.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Capability params as strings (`temperature`, `steps`, `guidance`,
    /// `width`, `voice`, `speed`, …). Adapters parse what they support and
    /// ignore the rest, so unknown capabilities degrade to a generic form
    /// instead of crashing.
    #[serde(default)]
    pub params: HashMap<String, String>,
    /// Reproducibility seed, if the form exposes one.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Where output files go; defaults to `outputs/<Kind>/` under data_dir.
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
}

impl InferenceRequest {
    pub fn new(model: ModelRecord, model_dir: PathBuf, prompt: String) -> Self {
        Self {
            model,
            model_dir,
            prompt,
            negative_prompt: None,
            params: HashMap::new(),
            seed: None,
            output_dir: None,
        }
    }
}

/// What comes back from `run`. Every result carries enough metadata to write
/// the reproducibility sidecar (`docs/04-DATA-MODEL.md` §4.5).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Adapter that produced this result (`transformers`, `diffusers`, …).
    pub runtime_id: String,
    /// Generated text, if the capability produces any.
    #[serde(default)]
    pub text: Option<String>,
    /// Generated files (images, audio, video).
    #[serde(default)]
    pub files: Vec<PathBuf>,
    /// Sidecar fields: model, prompt, seed, params, runtime, timestamp.
    #[serde(default)]
    pub sidecar: HashMap<String, String>,
    /// Wall-clock inference time.
    #[serde(default)]
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// Environment handles (adapter-side view).
// ---------------------------------------------------------------------------

/// Adapter-side view of an isolated env.
///
/// The source of truth for env lifecycle (create / shared-reuse resolver /
/// pin record / rollback) is [`EnvManager`]; this is the resolved path set
/// adapters spawn child processes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvRef {
    pub id: String,
    pub dir: PathBuf,
    pub python_exe: PathBuf,
    pub environment_json: PathBuf,
}

impl From<EnvHandle> for EnvRef {
    fn from(h: EnvHandle) -> Self {
        Self {
            id: h.id,
            dir: h.dir,
            python_exe: h.python_exe,
            environment_json: h.environment_json,
        }
    }
}

/// Resolve the conventional default env for a runtime kind
/// (`environments/<runtime-id>`).
///
/// Convention-path fallback, kept for `install()` (which must create a
/// *specific* dir for `bootstrap.py --env-dir`) and `health_check()`.
/// Inference paths ([`prepare_common`]) go through
/// [`EnvManager::resolve_for_runtime`] so compatible models share one env
/// (`docs/09-ENV-SECURITY.md` §9.1).
pub fn default_env_ref(data_dir: &Path, runtime_id: &str) -> EnvRef {
    let dir = data_dir.join("environments").join(runtime_id);
    EnvRef {
        python_exe: venv_python(&dir),
        environment_json: dir.join("environment.json"),
        id: runtime_id.to_string(),
        dir,
    }
}

/// Interpreter inside an isolated venv layout — never a global `python`.
#[cfg(windows)]
pub fn venv_python(env_dir: &Path) -> PathBuf {
    env_dir.join("python").join("Scripts").join("python.exe")
}

/// Interpreter inside an isolated venv layout — never a global `python`.
#[cfg(not(windows))]
pub fn venv_python(env_dir: &Path) -> PathBuf {
    env_dir.join("python").join("bin").join("python")
}

// ---------------------------------------------------------------------------
// Supervised child-process helpers.
// ---------------------------------------------------------------------------

/// Run a short-lived helper (`bootstrap.py`, `serve.py --job`, health
/// probes) to completion, capturing output. Spawn failures and non-zero
/// exits map to [`RuntimeError`] codes; the app itself is never at risk
/// from a runtime crash. `runtime` is the adapter id for crash attribution.
pub fn supervised_command(
    program: &Path,
    args: &[String],
    cwd: &Path,
    runtime: &str,
) -> RuntimeResult<std::process::Output> {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| {
            RuntimeError::new(
                codes::CHILD_SPAWN_FAILED,
                format!("could not launch {}: {e}", program.display()),
                "Reinstall the runtime env from Environments, then retry. If it persists, \
                 check antivirus quarantine — fresh venv interpreters are sometimes flagged.",
            )
            .with_runtime(runtime)
        })?;
    if output.status.success() {
        Ok(output)
    } else {
        let tail = String::from_utf8_lossy(&output.stderr);
        let tail: String = tail
            .chars()
            .rev()
            .take(800)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        Err(RuntimeError::crashed(
            format!("helper {} exited with {}", program.display(), output.status),
            tail.trim(),
        )
        .with_runtime(runtime))
    }
}

/// Shared `prepare()` prologue: model dir must exist and the isolated env
/// (via [`EnvManager::resolve_for_runtime`], so compatible models share one
/// env) must contain an interpreter. Returns the resolved [`EnvRef`].
pub fn prepare_common(
    data_dir: &Path,
    runtime_id: &str,
    model_dir: &Path,
) -> RuntimeResult<EnvRef> {
    if !model_dir.is_dir() {
        return Err(RuntimeError::model_not_found(model_dir).with_runtime(runtime_id));
    }
    let manager = EnvManager::new(data_dir.to_path_buf());
    let handle = manager.resolve_for_runtime(runtime_id);
    if !handle.python_exe.is_file() {
        return Err(RuntimeError::env_missing(&handle.id, &handle.dir).with_runtime(runtime_id));
    }
    Ok(EnvRef::from(handle))
}

/// Shared `run()` prologue for Python-backed adapters: resolves the env and
/// locates the serve entrypoint under `runtimes/<id>/serve.py`.
pub fn require_serve_entrypoint(
    data_dir: &Path,
    runtime_id: &str,
    env: &EnvRef,
) -> RuntimeResult<PathBuf> {
    let _ = env;
    let entry = data_dir.join("runtimes").join(runtime_id).join("serve.py");
    if !entry.is_file() {
        return Err(RuntimeError::new(
            codes::RUNTIME_NOT_READY,
            format!("serve entrypoint not bundled yet: {}", entry.display()),
            "This runtime's executor script has not landed in this build. Track it in \
             docs/12-BUILD-TASKS.md Phase C/D; env setup (bootstrap.py) already works.",
        )
        .with_runtime(runtime_id));
    }
    Ok(entry)
}

/// Job file handed to `serve.py --job` (one-shot) or `POST /run`
/// (persistent server). Field names are the contract — `serve.py` reads the
/// same keys, so never rename without updating every `runtimes/*/serve.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobPayload {
    pub runtime: String,
    pub model_id: String,
    pub repository: String,
    pub revision: String,
    pub model_dir: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub params: HashMap<String, String>,
}

/// Serialize the job file handed to `serve.py` via argv/HTTP.
pub fn job_json(runtime_id: &str, req: &InferenceRequest) -> String {
    let payload = JobPayload {
        runtime: runtime_id.to_string(),
        model_id: req.model.id.clone(),
        repository: req.model.repository.clone(),
        revision: req
            .model
            .revision
            .clone()
            .unwrap_or_else(|| "main".to_string()),
        model_dir: req.model_dir.to_string_lossy().to_string(),
        prompt: req.prompt.clone(),
        negative_prompt: req.negative_prompt.clone().unwrap_or_default(),
        seed: req.seed,
        params: req.params.clone(),
    };
    serde_json::to_string(&payload)
        .unwrap_or_else(|_| "{\"runtime\":\"job-serialize-failed\"}".to_string())
}

/// Parse a job file back (symmetric with [`job_json`]; used by tests and by
/// future Rust-side persistent-server clients).
pub fn parse_job(text: &str) -> Result<JobPayload, serde_json::Error> {
    serde_json::from_str(text)
}

// ---------------------------------------------------------------------------
// The trait. Every runtime implements exactly this; the UI calls
// `prepare` → `run` and nothing else backend-specific.
// ---------------------------------------------------------------------------

/// `docs/06-RUNTIME-ADAPTERS.md` §6.1, plus `id()`/`label()`.
///
/// `id()` (extension over the doc trait) is required so the analyzer registry
/// (`runtimes/registry.json`) and the env manager can address adapters
/// without `match` arms on concrete types.
///
/// `model` is the canonical [`ModelRecord`]; `model_dir` is its on-disk
/// snapshot dir (`models/<id>`), resolved by the caller because the record
/// carries no path.
pub trait RuntimeAdapter: Send + Sync {
    /// Registry + env key, e.g. `"transformers"`, `"diffusers"`, `"llama_cpp"`.
    fn id(&self) -> &'static str;
    /// Human label for the Runtime page, e.g. `"Transformers (PyTorch)"`.
    fn label(&self) -> &'static str;
    /// Pure file/metadata inspection — must never spawn processes or touch
    /// the network. Multiple adapters may claim one model; the analyzer
    /// picks via HW compat (`docs/05-HF-INTEGRATION.md` §5.5).
    fn detect(&self, model: &ModelRecord, model_dir: &Path) -> bool;
    /// Create/refresh the isolated env (drives `bootstrap.py` as a child
    /// process). Idempotent: safe to call when already installed.
    fn install(&self) -> RuntimeResult<()>;
    /// Validate env + model and stage anything `run` needs (weights check,
    /// server warm-up, job staging). UI calls this first, always.
    fn prepare(&self, model: &ModelRecord, model_dir: &Path) -> RuntimeResult<()>;
    /// Execute one inference request in a supervised child process and
    /// return the result with its reproducibility sidecar.
    fn run(&self, req: InferenceRequest) -> RuntimeResult<InferenceResult>;
    /// Stop a persistent server child process, if any. No-op for one-shot
    /// executors. Must never fail when nothing is running.
    fn stop(&self) -> RuntimeResult<()>;
    /// Liveness probe: running server wins; otherwise the isolated
    /// interpreter must exist.
    fn health_check(&self) -> RuntimeResult<()>;
}

/// Base state shared by every adapter: data-dir root + an optional
/// persistent server child. Embed and delegate (`stop_child`,
/// `child_running`, `take_child_output` as needed).
pub struct AdapterBase {
    data_dir: PathBuf,
    child: Mutex<Option<Child>>,
}

impl AdapterBase {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            child: Mutex::new(None),
        }
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Remember a spawned server child so `stop()` / `health_check()` can
    /// manage it. Previous child, if any, is left alone (replaced).
    pub fn track(&self, child: Child) {
        if let Ok(mut slot) = self.child.lock() {
            *slot = Some(child);
        }
    }

    /// Kill the tracked server child, if any. Always succeeds when nothing
    /// is running.
    pub fn stop_child(&self) -> RuntimeResult<()> {
        let mut slot = self.child.lock().map_err(|_| {
            RuntimeError::new(
                codes::RUNTIME_CRASHED,
                "adapter child-process lock poisoned",
                "Restart the app; report this — it means a runtime thread panicked.",
            )
        })?;
        if let Some(mut child) = slot.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        Ok(())
    }

    /// True while a tracked server child is still alive. A dead child is
    /// reaped and forgotten (its exit is reported by `run`, not here).
    pub fn child_running(&self) -> bool {
        let mut slot = match self.child.lock() {
            Ok(s) => s,
            Err(_) => return false,
        };
        match slot.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(None) => true,
                _ => {
                    *slot = None;
                    false
                }
            },
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ModelState;

    fn record(id: &str) -> ModelRecord {
        ModelRecord {
            id: id.to_string(),
            name: id.to_string(),
            repository: id.to_string(),
            revision: None,
            task: None,
            runtime: None,
            size_bytes: None,
            status: ModelState::Discovered,
            capabilities: vec![],
            license: None,
            trust_level: None,
            est_weights_bytes: None,
            est_kv_bytes: None,
            est_total_bytes: None,
        }
    }

    #[test]
    fn job_payload_serde_roundtrip() {
        let mut params = HashMap::new();
        params.insert("temperature".to_string(), "0.7".to_string());
        let req = InferenceRequest {
            model: record("owner/model"),
            model_dir: PathBuf::from("models/owner-model"),
            prompt: "hello".to_string(),
            negative_prompt: Some("blurry".to_string()),
            params,
            seed: Some(42),
            output_dir: None,
        };
        let text = job_json("transformers", &req);
        let back = parse_job(&text).expect("job json parses");
        assert_eq!(back.runtime, "transformers");
        assert_eq!(back.model_id, "owner/model");
        assert_eq!(back.revision, "main");
        assert_eq!(back.seed, Some(42));
        assert_eq!(
            back.params.get("temperature").map(String::as_str),
            Some("0.7")
        );
    }

    #[test]
    fn stable_codes_survive_nexora_mapping() {
        let e = RuntimeError::env_missing("transformers", Path::new("environments/transformers"));
        assert_eq!(e.stable_code(), codes::ENV_MISSING);
        let n = NexoraError::from(e);
        let shown = format!("{n}");
        assert!(shown.contains("E-ENV-MISSING"), "stable code kept: {shown}");
        assert!(!n.human_fix().is_empty(), "every error carries a human fix");
    }

    #[test]
    fn cuda_oom_maps_to_cuda_variant_with_hint() {
        let e = RuntimeError::crashed(
            "serve.py failed",
            "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2GB",
        );
        assert!(looks_like_cuda_oom(&e.message));
        let n = NexoraError::from(e);
        assert_eq!(n.code(), "E_CUDA_OOM");
        assert!(n.human_fix().contains("FP16"), "CUDA-OOM human hint kept");
    }

    #[test]
    fn plain_crash_maps_to_runtime_crash_with_id() {
        let e = RuntimeError::crashed("segfault", "boom").with_runtime("diffusers");
        let n = NexoraError::from(e);
        assert_eq!(n.code(), "E_RUNTIME_CRASH");
        assert!(format!("{n}").contains("diffusers"));
        assert!(format!("{n}").contains("E-RUNTIME-CRASHED"));
    }

    #[test]
    fn custom_code_block_maps_with_consent_hint() {
        let e = RuntimeError::new(
            codes::CUSTOM_CODE_BLOCKED,
            "model 'x/y' ships custom code (custom_model.py)",
            "Open the trust prompt: [View Files] to inspect the code.",
        );
        let n = NexoraError::from(e);
        assert_eq!(n.code(), "E_CUSTOM_CODE");
        let shown = format!("{n}");
        assert!(shown.contains("E-CUSTOM-CODE-BLOCKED"));
        assert!(shown.contains("[Run in Sandbox]") || shown.contains("[View Files]"));
    }

    #[test]
    fn comfyui_routing_is_flagged_before_conversion() {
        let e = RuntimeError::new(
            codes::USE_COMFYUI,
            "video generation is delegated to the ComfyUI adapter",
            "Install the ComfyUI runtime from Environments, then generate again.",
        );
        assert!(e.is_routing());
    }
}
