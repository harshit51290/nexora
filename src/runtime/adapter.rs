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
//! * Each runtime owns an **isolated** interpreter at
//!   `environments/<runtime-id>/…`. Global `pip` is never touched.
//! * Every failure is a [`RuntimeError`] carrying a machine-readable `code`
//!   plus a human `hint` (`docs/11-UI-UX.md` §11.6 — no bare stack traces).
//!
//! TODO-CORE-ALIGN: once `src/core` / `src/model` land, replace the local
//! [`Model`] with the canonical core type, replace [`RuntimeError`] with the
//! core error catalog, and derive `serde::Serialize/Deserialize` on
//! [`InferenceRequest`] / [`InferenceResult`] / [`EnvPin`].

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Mutex;

/// Stable, machine-readable error codes surfaced to the UI, logs, and API.
/// Each code maps to a human cause + fix (see [`RuntimeError::hint`]).
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
    /// complex graphs). Not a failure — a routing instruction.
    pub const USE_COMFYUI: &str = "E-USE-COMFYUI";
    /// Estimated VRAM exceeds the active hardware profile (4 GB reference).
    pub const VRAM_LOW: &str = "E-VRAM-LOW";
}

/// A single runtime failure: machine code + technical message + human fix.
///
/// `scope` is the log scope from `docs/11-UI-UX.md` §11.6
/// (`Runtime` / `Download` / `Model` / `System`).
#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub code: &'static str,
    pub message: String,
    pub hint: String,
    pub scope: &'static str,
}

impl RuntimeError {
    pub fn new(
        code: &'static str,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            hint: hint.into(),
            scope: "Runtime",
        }
    }

    pub fn env_missing(env_id: &str, env_dir: &Path) -> Self {
        Self::new(
            codes::ENV_MISSING,
            format!("isolated env '{env_id}' has no interpreter at {}", env_dir.display()),
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
        let hint = if stderr_tail.contains("out of memory")
            || stderr_tail.contains("CUDA error")
            || stderr_tail.contains("allocation")
        {
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

/// Convenience alias used by every adapter method.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

// ---------------------------------------------------------------------------
// TODO-CORE-ALIGN: minimal model descriptor.
// Replace with `crate::core::Model` (SQLite `models` row, docs/04-DATA-MODEL)
// once `src/core` / `src/model` exist. Field names intentionally mirror the
// `models` table plus analyzer outputs (`docs/05-HF-INTEGRATION.md` §5.2).
// ---------------------------------------------------------------------------

/// Minimal model descriptor — just enough for `detect()` and `prepare()`.
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// `models.id` (SQLite primary key).
    pub id: String,
    /// `owner/repo` on Hugging Face.
    pub repository: String,
    /// Pinned revision (`@rev`); `None` = main.
    pub revision: Option<String>,
    /// Local snapshot dir holding `config.json` / weights / tokenizer files.
    pub local_dir: PathBuf,
    /// `config.json → architectures` (e.g. `["LlamaForCausalLM"]`).
    pub architectures: Vec<String>,
    /// `config.json → model_type` (e.g. `"llama"`).
    pub model_type: Option<String>,
    /// HF `pipeline_tag` (e.g. `"text-generation"`).
    pub pipeline_tag: Option<String>,
    /// HF `library_name` (e.g. `"transformers"`, `"diffusers"`).
    pub library_name: Option<String>,
    /// Capability tokens (`docs/04-DATA-MODEL.md` §4.4).
    pub capabilities: Vec<String>,
    /// Trust level (`docs/09-ENV-SECURITY.md`): Trusted | Community |
    /// Unverified | Blocked.
    pub trust_level: String,
}

impl Model {
    /// True when `local_dir/<name>` exists (e.g. `config.json`,
    /// `model_index.json`, `custom_model.py`).
    pub fn has_file(&self, name: &str) -> bool {
        self.local_dir.join(name).exists()
    }

    /// True when any file directly under `local_dir` (or one level of
    /// subdirectories, covering HF `blobs/` layouts) ends with `ext`.
    pub fn has_extension(&self, ext: &str) -> bool {
        has_extension_in(&self.local_dir, ext, 2)
    }

    /// Read a small metadata file next to the weights, if present.
    pub fn read_meta(&self, name: &str) -> Option<String> {
        std::fs::read_to_string(self.local_dir.join(name)).ok()
    }
}

fn has_extension_in(dir: &Path, ext: &str, depth: u32) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
            if path.extension().and_then(|e| e.to_str()) == Some(ext.trim_start_matches('.')) {
                return true;
            }
        } else if path.is_dir() && depth > 0 {
            if has_extension_in(&path, ext, depth - 1) {
                return true;
            }
        }
    }
    false
}

// --- tiny serde-free JSON helpers ------------------------------------------
// The analyzer metadata files (`config.json`, `model_index.json`) are read
// with plain string scans so this crate stays dependency-free.
// TODO-CORE-ALIGN: replace with `serde_json` once core adds it.

/// Extract the first string value for `"key": "value"` in a JSON document.
pub fn json_string(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut rest = text;
    loop {
        let i = rest.find(&needle)?;
        let after = rest[i + needle.len()..].trim_start();
        let after = after.strip_prefix(':')?.trim_start();
        if let Some(quoted) = after.strip_prefix('"') {
            let end = quoted.find('"')?;
            return Some(quoted[..end].to_string());
        }
        rest = &rest[i + needle.len()..];
    }
}

/// Extract the string array for `"key": ["a", "b"]` in a JSON document.
pub fn json_string_array(text: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\"");
    let i = match text.find(&needle) {
        Some(i) => i,
        None => return Vec::new(),
    };
    let after = text[i + needle.len()..].trim_start();
    let after = match after.strip_prefix(':') {
        Some(a) => a.trim_start(),
        None => return Vec::new(),
    };
    let inner = match after.strip_prefix('[') {
        Some(a) => a,
        None => return Vec::new(),
    };
    let end = match inner.find(']') {
        Some(e) => e,
        None => return Vec::new(),
    };
    inner[..end]
        .split(',')
        .filter_map(|item| {
            let item = item.trim().trim_matches('"').trim().to_string();
            if item.is_empty() {
                None
            } else {
                Some(item)
            }
        })
        .collect()
}

/// Minimal JSON string escaping for the job files we hand to child processes.
pub fn json_escape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 2);
    for c in raw.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Inference types (UI builds these; adapters consume them).
// ---------------------------------------------------------------------------

/// One generation request. The UI builds this from the capability-driven
/// form (`docs/11-UI-UX.md` §11.4) and calls `prepare` → `run`.
#[derive(Debug, Clone, Default)]
pub struct InferenceRequest {
    pub model: Model,
    /// Main prompt / input text.
    pub prompt: String,
    /// Diffusers-style negative prompt, if the capability form provides one.
    pub negative_prompt: Option<String>,
    /// Capability params as strings (`temperature`, `steps`, `guidance`,
    /// `width`, `voice`, `speed`, …). Adapters parse what they support and
    /// ignore the rest, so unknown capabilities degrade to a generic form
    /// instead of crashing.
    pub params: HashMap<String, String>,
    /// Reproducibility seed, if the form exposes one.
    pub seed: Option<u64>,
    /// Where output files go; defaults to `outputs/<Kind>/` under data_dir.
    pub output_dir: Option<PathBuf>,
}

/// What comes back from `run`. Every result carries enough metadata to write
/// the reproducibility sidecar (`docs/04-DATA-MODEL.md` §4.5).
#[derive(Debug, Clone, Default)]
pub struct InferenceResult {
    /// Adapter that produced this result (`transformers`, `diffusers`, …).
    pub runtime_id: String,
    /// Generated text, if the capability produces any.
    pub text: Option<String>,
    /// Generated files (images, audio, video).
    pub files: Vec<PathBuf>,
    /// Sidecar fields: model, prompt, seed, params, runtime, timestamp.
    pub sidecar: HashMap<String, String>,
    /// Wall-clock inference time.
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// Environment handles (adapter-side view).
// ---------------------------------------------------------------------------

/// Adapter-side view of an isolated env.
///
/// The source of truth for env lifecycle (create / shared-reuse resolver /
/// pin record / rollback) is `crate::env::EnvManager`; adapters only resolve
/// the conventional default `environments/<runtime-id>` path (MVP starts
/// 1-env-per-runtime-kind, `docs/09-ENV-SECURITY.md` §9.1) and verify the
/// interpreter exists before spawning anything.
#[derive(Debug, Clone)]
pub struct EnvRef {
    pub id: String,
    pub dir: PathBuf,
    pub python_exe: PathBuf,
    pub environment_json: PathBuf,
}

/// Resolve the conventional default env for a runtime kind.
/// TODO-CORE-ALIGN: ask `EnvManager::resolve_shared_env` instead once wired.
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

/// Run a short-lived helper (`bootstrap.py`, health probes) to completion,
/// capturing output. Spawn failures and non-zero exits map to [`RuntimeError`]
/// codes; the app itself is never at risk from a runtime crash.
pub fn supervised_command(
    program: &Path,
    args: &[String],
    cwd: &Path,
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
        })?;
    if output.status.success() {
        Ok(output)
    } else {
        let tail = String::from_utf8_lossy(&output.stderr);
        let tail: String = tail.chars().rev().take(800).collect::<String>().chars().rev().collect();
        Err(RuntimeError::crashed(
            format!("helper {} exited with {}", program.display(), output.status),
            tail.trim(),
        ))
    }
}

/// Shared `prepare()` prologue: model dir must exist and the isolated env
/// must contain an interpreter. Returns the resolved [`EnvRef`].
pub fn prepare_common(data_dir: &Path, runtime_id: &str, model: &Model) -> RuntimeResult<EnvRef> {
    if !model.local_dir.is_dir() {
        return Err(RuntimeError::model_not_found(&model.local_dir));
    }
    let env = default_env_ref(data_dir, runtime_id);
    if !env.python_exe.is_file() {
        return Err(RuntimeError::env_missing(&env.id, &env.dir));
    }
    Ok(env)
}

/// Shared `run()` prologue for Python-backed adapters: resolves the env and
/// locates the serve entrypoint under `runtimes/<id>/serve.py`.
///
/// MVP scaffold note: only `bootstrap.py` + `environment.json` stubs ship so
/// far, so this returns `E-RUNTIME-NOT-READY` until the per-runtime `serve.py`
/// lands. The plumbing (job file → spawn → crash mapping) is already real.
pub fn require_serve_entrypoint(data_dir: &Path, runtime_id: &str, env: &EnvRef) -> RuntimeResult<PathBuf> {
    let _ = env;
    let entry = data_dir
        .join("runtimes")
        .join(runtime_id)
        .join("serve.py");
    if !entry.is_file() {
        return Err(RuntimeError::new(
            codes::RUNTIME_NOT_READY,
            format!("serve entrypoint not bundled yet: {}", entry.display()),
            "This runtime's executor script has not landed in this build. Track it in \
             docs/12-BUILD-TASKS.md Phase C/D; env setup (bootstrap.py) already works.",
        ));
    }
    Ok(entry)
}

/// Serialize the job file handed to `serve.py` via stdin/argv.
pub fn job_json(runtime_id: &str, req: &InferenceRequest) -> String {
    let mut params: Vec<String> = req
        .params
        .iter()
        .map(|(k, v)| format!("\"{}\":\"{}\"", json_escape(k), json_escape(v)))
        .collect();
    params.sort();
    format!(
        "{{\"runtime\":\"{}\",\"model_id\":\"{}\",\"repository\":\"{}\",\"revision\":\"{}\",\
         \"model_dir\":\"{}\",\"prompt\":\"{}\",\"negative_prompt\":\"{}\",\"seed\":{},\
         \"params\":{{{}}}}}",
        json_escape(runtime_id),
        json_escape(&req.model.id),
        json_escape(&req.model.repository),
        json_escape(req.model.revision.as_deref().unwrap_or("main")),
        json_escape(&req.model.local_dir.to_string_lossy()),
        json_escape(&req.prompt),
        json_escape(req.negative_prompt.as_deref().unwrap_or("")),
        req.seed.map(|s| s.to_string()).unwrap_or_else(|| "null".to_string()),
        params.join(","),
    )
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
pub trait RuntimeAdapter: Send + Sync {
    /// Registry + env key, e.g. `"transformers"`, `"diffusers"`, `"llama_cpp"`.
    fn id(&self) -> &'static str;
    /// Human label for the Runtime page, e.g. `"Transformers (PyTorch)"`.
    fn label(&self) -> &'static str;
    /// Pure file/metadata inspection — must never spawn processes or touch
    /// the network. Multiple adapters may claim one model; the analyzer
    /// picks via HW compat (`docs/05-HF-INTEGRATION.md` §5.5).
    fn detect(&self, model: &Model) -> bool;
    /// Create/refresh the isolated env (drives `bootstrap.py` as a child
    /// process). Idempotent: safe to call when already installed.
    fn install(&self) -> RuntimeResult<()>;
    /// Validate env + model and stage anything `run` needs (weights check,
    /// server warm-up, job staging). UI calls this first, always.
    fn prepare(&self, model: &Model) -> RuntimeResult<()>;
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
