//! Shared API surface: DTOs, structured errors, core facade, router.
//!
//! Docs: `docs/10-API-CLI.md` §10.1–10.2, `docs/04-DATA-MODEL.md` §4.4–4.5,
//! global gate in `docs/12-BUILD-TASKS.md` (every ERROR: code + human fix).

pub mod migrations;
pub mod openai;
pub mod rest;
pub mod service;
pub mod versions;
pub mod ws;

use axum::{http::StatusCode, response::{IntoResponse, Json}, Router};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// Shared state for all routers (Axum 0.7: provided via `.with_state`).
#[derive(Clone, Debug, Default)]
pub struct AppState {
    /// Crate version serving the API; surfaced in error bodies for triage.
    pub version: &'static str,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
        }
    }
}

// ---------------------------------------------------------------------------
// Structured errors (code + human fix, per global gate)
// ---------------------------------------------------------------------------

/// Machine-readable error envelope returned by every endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
    pub fix: String,
    /// Crate version that produced the error.
    pub version: String,
}

pub struct ApiError {
    pub status: StatusCode,
    pub body: ErrorBody,
}

impl ApiError {
    pub fn from_stub(version: &'static str, e: core_stub::CoreStubError) -> Self {
        Self::from_core(version, e)
    }

    /// Status comes from the machine-readable code so every failure keeps
    /// the `ErrorBody {code,message,fix}` shape with the right status.
    pub fn from_core(version: &'static str, e: core_stub::CoreStubError) -> Self {
        let status = match e.code {
            "E_MODEL_NOT_FOUND" | "E_RUNTIME_NOT_FOUND" => StatusCode::NOT_FOUND,
            "E_BAD_HF_URL" | "E_EMPTY_PROMPT" | "E_EMPTY_MESSAGES" | "E_BAD_DOWNLOAD_ACTION"
            | "E_MODEL_NOT_READY" | "E_MODEL_UNSUPPORTED" | "E_CAPABILITY_MISMATCH"
            | "E_INSTALL_FAILED" => StatusCode::BAD_REQUEST,
            "E_CUSTOM_CODE" => StatusCode::FORBIDDEN,
            "E_CORE_NOT_WIRED" | "E_STREAMING_DEFERRED" => StatusCode::NOT_IMPLEMENTED,
            "E_VRAM_SHORT" | "E_CUDA_OOM" | "E_SPACE_LOW" => StatusCode::INSUFFICIENT_STORAGE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            body: ErrorBody {
                code: e.code.into(),
                message: e.message,
                fix: e.fix,
                version: version.into(),
            },
        }
    }

    pub fn bad_request(
        version: &'static str,
        code: &'static str,
        message: String,
        fix: &'static str,
    ) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorBody {
                code: code.into(),
                message,
                fix: fix.into(),
                version: version.into(),
            },
        }
    }

    pub fn unimplemented(
        version: &'static str,
        code: &'static str,
        message: String,
        fix: &'static str,
    ) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            body: ErrorBody {
                code: code.into(),
                message,
                fix: fix.into(),
                version: version.into(),
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(self.body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

/// Row of the `models` table (docs/04 §4.1), trimmed for list views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub repository: String,
    /// Model state machine status (docs/04 §4.2).
    pub status: String,
    /// Capability tokens (docs/04 §4.4), e.g. `["text-to-image"]`.
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSummary {
    pub id: String,
    /// MVP: `transformers` | `diffusers` | `llama.cpp`.
    pub kind: String,
    pub version: String,
    /// Runtime state machine status (docs/04 §4.3).
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSummary {
    pub label: String,
    pub gpu_vram_gb: Option<f32>,
    pub system_ram_gb: Option<f32>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallRequest {
    /// HF repo id (`owner/model`) or full URL (`https://huggingface.co/...`).
    pub repo: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallResponse {
    pub model_id: String,
    /// Next state-machine status, e.g. `DOWNLOADING` (docs/04 §4.2).
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadRequest {
    pub model_id: String,
}

/// `POST /models/estimate` (docs/10 §10.1, hf-mem method via `src/mem`):
/// weight/KV bytes WITHOUT downloading. All fields optional except `repo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimateRequest {
    /// HF repo id (`owner/model`) or full URL (`https://huggingface.co/...`).
    pub repo: String,
    pub revision: Option<String>,
    #[serde(default)]
    pub experimental: bool,
    #[serde(default)]
    pub max_model_len: Option<u64>,
    #[serde(default = "default_batch_size")]
    pub batch_size: u64,
    #[serde(default = "default_kv_dtype")]
    pub kv_cache_dtype: String,
    #[serde(default)]
    pub gguf_file: Option<String>,
}

fn default_batch_size() -> u64 {
    1
}

fn default_kv_dtype() -> String {
    "auto".into()
}

/// `POST /generate` / `POST /v1/generate {model,prompt,width,height}`
/// (docs/10 §10.1). Runner picks the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: Option<u64>,
    pub params: Option<serde_json::Value>,
    /// VRAM-gate overrides (`uar run/generate --cpu/--offload/--gpu-layers`).
    /// `None` (and missing in REST JSON) = default gating: warn-and-proceed
    /// inside the CPU-offload window, hard-fail past it (`E_VRAM_SHORT`).
    #[serde(default)]
    pub execution: Option<ExecutionPrefs>,
}

/// Execution placement overrides for one generate/load.
///
/// * `cpu`: skip the VRAM gate, run fully on CPU.
/// * `offload`: widen the gate — proceed with CPU offload even past the
///   normal offload window instead of hard-failing.
/// * `gpu_layers`: layers to keep on GPU when offloading (llama.cpp style);
///   forwarded to the adapter as the `n_gpu_layers` param.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionPrefs {
    #[serde(default)]
    pub cpu: bool,
    #[serde(default)]
    pub offload: bool,
    #[serde(default)]
    pub gpu_layers: Option<u32>,
}

/// Output sidecar (docs/04 §4.5): every file under `outputs/...` ships one
/// of these as `<output>.json` (model, prompt, seed, params, timestamp,
/// runtime — powers history Reuse/Regenerate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSidecar {
    pub model: String,
    pub prompt: String,
    pub seed: Option<u64>,
    pub params: serde_json::Value,
    pub timestamp: String,
    pub runtime: String,
}

impl GenerationSidecar {
    pub fn new(
        model: String,
        prompt: String,
        seed: Option<u64>,
        params: serde_json::Value,
        runtime: String,
    ) -> Self {
        Self {
            model,
            prompt,
            seed,
            params,
            timestamp: chrono::Utc::now().to_rfc3339(),
            runtime,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub id: String,
    pub output_path: String,
    /// Inline text for text-generation models; `None` for media outputs
    /// (those are referenced by `output_path` + `sidecar`).
    pub text: Option<String>,
    pub sidecar: GenerationSidecar,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Combined router: REST (§10.1) + OpenAI-compat (§10.2) + WS stub (§10.2).
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .merge(rest::router(state.clone()))
        .merge(openai::router(state.clone()))
        .merge(ws::router(state))
}

// ---------------------------------------------------------------------------
// Core facade stubs
// ---------------------------------------------------------------------------

/// Facade over the Rust core (`model_manager`, `hardware`, `download`,
/// `scheduler`, `runtime` adapters per docs/02 §2.4).
///
/// Every function below calls the REAL manager via [`service::core`], EXCEPT
/// the pure helpers (`parse_hf_url`, `classify_model`) which are local logic
/// pinned by `tests/mvp_flow.rs`. Failures surface as [`CoreStubError`]
/// (`code + message + fix`, per the global gate) and map to HTTP status in
/// [`ApiError::from_core`].
pub mod core_stub {
    use super::{
        ExecutionPrefs, GenerateRequest, GenerationResult, HardwareSummary, ModelSummary,
        RuntimeSummary,
    };
    use crate::core::NexoraError;
    use crate::model::manager::ModelRecord;
    use crate::runtime::RuntimeError;

    /// Fix text shared by every not-wired error (global gate: human fix).
    pub const NOT_WIRED_FIX: &str = "Manager exists but is not wired to this endpoint yet (TODO-WIRE). Track docs/12-BUILD-TASKS.md Phase C; pure helpers (URL parse, classify) already work.";

    #[derive(Debug, Clone, thiserror::Error)]
    #[error("[{code}] {message} (fix: {fix})")]
    pub struct CoreStubError {
        pub code: &'static str,
        pub message: String,
        /// Human fix. `String` (not `&'static str`) so manager errors
        /// (`NexoraError::human_fix`, `RuntimeError::hint`) survive.
        pub fix: String,
    }

    impl CoreStubError {
        pub fn coded(
            code: &'static str,
            message: impl Into<String>,
            fix: impl Into<String>,
        ) -> Self {
            Self {
                code,
                message: message.into(),
                fix: fix.into(),
            }
        }
    }

    impl From<NexoraError> for CoreStubError {
        fn from(e: NexoraError) -> Self {
            Self {
                code: e.code(),
                message: e.to_string(),
                fix: e.human_fix(),
            }
        }
    }

    impl From<RuntimeError> for CoreStubError {
        fn from(e: RuntimeError) -> Self {
            Self {
                code: e.code,
                message: e.message,
                fix: e.hint,
            }
        }
    }

    impl From<sqlx::Error> for CoreStubError {
        fn from(e: sqlx::Error) -> Self {
            Self::from(NexoraError::Db(e))
        }
    }

    impl From<std::io::Error> for CoreStubError {
        fn from(e: std::io::Error) -> Self {
            Self::from(NexoraError::Io(e))
        }
    }

    pub fn not_wired(feature: &'static str, detail: String) -> CoreStubError {
        CoreStubError {
            code: "E_CORE_NOT_WIRED",
            message: format!("TODO-CORE-WIRE: {feature} not wired to core yet. {detail}"),
            fix: NOT_WIRED_FIX.to_string(),
        }
    }

    // -- pure helpers: REAL logic, keep when wiring -------------------------

    /// Accept a bare `owner/model` id or a full HF URL and return the
    /// canonical `owner/model` id. Tolerates `.git` suffix and
    /// `/blob/...` / `/tree/...` deep links.
    pub fn parse_hf_url(input: &str) -> Result<String, CoreStubError> {
        let t = input.trim();
        if t.is_empty() {
            return Err(CoreStubError {
                code: "E_BAD_HF_URL",
                message: "Empty model reference.".into(),
                fix: "Pass a repo id (owner/model) or URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.".to_string(),
            });
        }
        // Full URL form.
        if t.contains("://") {
            let path = t
                .trim_end_matches('/')
                .trim_end_matches(".git")
                .trim_end_matches('/');
            let rest = path
                .strip_prefix("https://huggingface.co/")
                .or_else(|| path.strip_prefix("http://huggingface.co/"))
                .or_else(|| path.strip_prefix("https://hf.co/"))
                .or_else(|| path.strip_prefix("http://hf.co/"))
                .ok_or_else(|| CoreStubError {
                    code: "E_BAD_HF_URL",
                    message: format!("Unsupported model URL host: {t}"),
                    fix: "Use a huggingface.co (or hf.co) repo URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.".to_string(),
                })?;
            let mut parts = rest.split('/').filter(|s| !s.is_empty());
            match (parts.next(), parts.next()) {
                (Some(owner), Some(model)) => Ok(format!("{owner}/{model}")),
                _ => Err(CoreStubError {
                    code: "E_BAD_HF_URL",
                    message: format!("URL has no owner/model path: {t}"),
                    fix: "Use a full repo URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.".to_string(),
                }),
            }
        } else {
            // Bare `owner/model` form.
            let mut parts = t.split('/').filter(|s| !s.is_empty());
            match (parts.next(), parts.next(), parts.next()) {
                (Some(owner), Some(model), None) => Ok(format!("{owner}/{model}")),
                _ => Err(CoreStubError {
                    code: "E_BAD_HF_URL",
                    message: format!("Not an owner/model id: {t}"),
                    fix: "Pass a repo id (owner/model) or URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.".to_string(),
                }),
            }
        }
    }

    /// Pre-classifier heuristic mirroring the Phase-B gate (docs/12 B3):
    /// SD1.5 -> Diffusers, Qwen-GGUF -> llama.cpp, unknown -> unsupported.
    /// TODO-CORE-WIRE: replace with the analyzer (config/model_index/file
    /// detection + `registry.json` compat score, docs/05).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ModelRecommendation {
        /// Capability token (docs/04 §4.4).
        pub task: String,
        /// `transformers` | `diffusers` | `llama.cpp` | `unsupported`.
        pub runtime: String,
        pub notes: String,
    }

    pub fn classify_model(repo: &str) -> ModelRecommendation {
        let lower = repo.to_lowercase();
        if lower.contains("stable-diffusion")
            || lower.contains("sdxl")
            || lower.contains("diffusers")
        {
            ModelRecommendation {
                task: "text-to-image".into(),
                runtime: "diffusers".into(),
                notes: "Stable Diffusion family -> Diffusers adapter; 4GB low-VRAM config recommended (docs/03 victory-1).".into(),
            }
        } else if lower.contains("gguf") {
            ModelRecommendation {
                task: "text-generation".into(),
                runtime: "llama.cpp".into(),
                notes: "GGUF weights -> llama.cpp adapter; Q4 quant default (docs/03 victory-2).".into(),
            }
        } else if lower.contains("qwen")
            || lower.contains("llama")
            || lower.contains("mistral")
            || lower.contains("gpt2")
        {
            ModelRecommendation {
                task: "text-generation".into(),
                runtime: "transformers".into(),
                notes: "HF transformers checkpoint -> Transformers adapter (isolated env).".into(),
            }
        } else {
            // TODO-CORE-WIRE: analyzer emits an unsupported card with an
            // Experimental-run offer instead of a bare label (docs/12 B3).
            ModelRecommendation {
                task: "unknown".into(),
                runtime: "unsupported".into(),
                notes: "Architecture not recognized; analyzer will offer an Experimental run card.".into(),
            }
        }
    }

    // -- manager calls: REAL wiring via `service::core` ----------------------

    async fn core() -> Result<super::service::CoreHandle, CoreStubError> {
        super::service::core().await
    }

    /// `ModelManager` install + `DownloadManager` fetch (resume/pause/
    /// checksum, docs/08); returns the canonical `owner/model` id. Bytes
    /// move in a background task; the row is already `DOWNLOADING`.
    pub async fn install_model(
        repo: &str,
        revision: Option<&str>,
    ) -> Result<String, CoreStubError> {
        core().await?.install_model(repo, revision).await
    }

    /// `ModelManager::load` (READY -> LOADED), after the VRAM gate and an
    /// env/runtime readiness probe.
    pub async fn load_model(model_id: &str) -> Result<(), CoreStubError> {
        core().await?.load_model(model_id).await
    }

    /// [`load_model`] with VRAM-gate overrides (`--cpu/--offload`; see
    /// [`ExecutionPrefs`]). Default gating applies when prefs are default.
    pub async fn load_model_with(
        model_id: &str,
        prefs: &ExecutionPrefs,
    ) -> Result<(), CoreStubError> {
        core().await?.load_model_with(model_id, prefs).await
    }

    /// `ModelManager::unload` (LOADED -> READY) + best-effort runtime stop.
    pub async fn unload_model(model_id: &str) -> Result<(), CoreStubError> {
        core().await?.unload_model(model_id).await
    }

    /// `SELECT id,name,repository,status,capabilities FROM models` (docs/04 §4.1).
    pub async fn list_models() -> Result<Vec<ModelSummary>, CoreStubError> {
        core().await?.list_models().await
    }

    /// Persisted job-queue rows (DB truth, docs/10 §10.4). Used by `uar jobs`.
    pub async fn list_jobs() -> Result<Vec<crate::jobs::Job>, CoreStubError> {
        core().await?.list_jobs().await
    }

    /// Memory estimate WITHOUT downloading (hf-mem method, `src/mem`):
    /// weight/KV bytes via Hub Range requests. No DB, no state machine —
    /// safe to call on any repo id or URL.
    pub async fn estimate(
        req: &super::EstimateRequest,
    ) -> Result<crate::mem::MemEstimate, CoreStubError> {
        // URL, `owner/model`, or bare name (Hub alias, e.g. `gpt2`).
        let id = match parse_hf_url(&req.repo) {
            Ok(id) => id,
            Err(_)
                if !req.repo.contains('/') && !req.repo.contains("://") =>
            {
                crate::hf::resolve_bare_name(&req.repo).await?.id()
            }
            Err(e) => return Err(e),
        };
        let (owner, name) = id.split_once('/').unwrap_or(("", id.as_str()));
        let mut repo = crate::hf::HfRepo {
            owner: owner.to_string(),
            repo: name.to_string(),
            rev: "main".to_string(),
        };
        if let Some(rev) = req.revision.as_deref() {
            repo.rev = rev.to_string();
        }
        let opts = crate::mem::EstimateOpts {
            experimental: req.experimental,
            max_model_len: req.max_model_len,
            batch_size: req.batch_size,
            kv_cache_dtype: req.kv_cache_dtype.clone(),
            gguf_file: req.gguf_file.clone(),
        };
        crate::mem::estimate_repo(&repo, &opts).await.map_err(CoreStubError::from)
    }

    /// `HardwareBackend::detect` (NVIDIA over a CPU baseline; 4GB is the
    /// reference low-end, AGENTS.md rule 6).
    pub async fn hardware_info() -> Result<HardwareSummary, CoreStubError> {
        core().await?.hardware_info().await
    }

    /// `SELECT` over `runtimes`, overlaid with the MVP registry
    /// (transformers/diffusers/llama.cpp, docs/03 §3.1) for ids with no
    /// stored row yet.
    pub async fn list_runtimes() -> Result<Vec<RuntimeSummary>, CoreStubError> {
        core().await?.list_runtimes().await
    }

    /// `Scheduler` admission -> `RuntimeAdapter::prepare` -> `run`
    /// (AGENTS.md rule 3). Persists the generation row + output sidecar
    /// (docs/04 §4.1, §4.5) on success.
    pub async fn generate(req: &GenerateRequest) -> Result<GenerationResult, CoreStubError> {
        core().await?.generate(req).await
    }

    /// Pause / resume / cancel the active download (`DownloadManager`
    /// flags). Returns the new `downloads.state`.
    pub async fn download_control(action: &str) -> Result<String, CoreStubError> {
        core().await?.download_control(action).await
    }

    /// Stored model row (capabilities, runtime, status) for capability
    /// gating (e.g. OpenAI chat requires a text model).
    pub async fn model_record(model_id: &str) -> Result<ModelRecord, CoreStubError> {
        core().await?.model_record(model_id).await
    }
}
