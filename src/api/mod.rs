//! Shared API surface: DTOs, structured errors, core facade stubs, router.
//!
//! Docs: `docs/10-API-CLI.md` §10.1–10.2, `docs/04-DATA-MODEL.md` §4.4–4.5,
//! global gate in `docs/12-BUILD-TASKS.md` (every ERROR: code + human fix).

pub mod openai;
pub mod rest;
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

/// Machine-readable error envelope returned by every stub endpoint.
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
        Self::unimplemented(version, e.code, e.message, e.fix)
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

/// Placeholder for the Rust core (`model_manager`, `runtime_manager`,
/// `hardware_manager`, `download_manager`, `scheduler` per docs/02 §2.4).
///
/// TODO-WIRE: managers now live under `src/` (model, hardware, download,
/// scheduler, runtime, env) — replace each stub body below with the real
/// call. Until then every function returns `E_CORE_NOT_WIRED`, EXCEPT the
/// pure helpers (`parse_hf_url`, `classify_model`) which are real logic and
/// are pinned by `tests/mvp_flow.rs`.
pub mod core_stub {
    use super::{GenerateRequest, GenerationResult, HardwareSummary, ModelSummary, RuntimeSummary};

    /// Fix text shared by every not-wired error (global gate: human fix).
    pub const NOT_WIRED_FIX: &str = "Manager exists but is not wired to this endpoint yet (TODO-WIRE). Track docs/12-BUILD-TASKS.md Phase C; pure helpers (URL parse, classify) already work.";

    #[derive(Debug, Clone, thiserror::Error)]
    #[error("[{code}] {message} (fix: {fix})")]
    pub struct CoreStubError {
        pub code: &'static str,
        pub message: String,
        pub fix: &'static str,
    }

    pub fn not_wired(feature: &'static str, detail: String) -> CoreStubError {
        CoreStubError {
            code: "E_CORE_NOT_WIRED",
            message: format!("TODO-CORE-WIRE: {feature} not wired to core yet. {detail}"),
            fix: NOT_WIRED_FIX,
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
                fix: "Pass a repo id (owner/model) or URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.",
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
                    fix: "Use a huggingface.co (or hf.co) repo URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.",
                })?;
            let mut parts = rest.split('/').filter(|s| !s.is_empty());
            match (parts.next(), parts.next()) {
                (Some(owner), Some(model)) => Ok(format!("{owner}/{model}")),
                _ => Err(CoreStubError {
                    code: "E_BAD_HF_URL",
                    message: format!("URL has no owner/model path: {t}"),
                    fix: "Use a full repo URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.",
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
                    fix: "Pass a repo id (owner/model) or URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.",
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

    // -- manager calls: STUBS, all E_CORE_NOT_WIRED --------------------------

    /// TODO-CORE-WIRE: `model_manager::install` + `download_manager`
    /// (resume/pause/checksum, docs/08) then analyzer VALIDATING step.
    pub async fn install_model(
        repo: &str,
        revision: Option<&str>,
    ) -> Result<String, CoreStubError> {
        Err(not_wired(
            "install_model",
            format!(
                "would install repo={repo} revision={} via download_manager.",
                revision.unwrap_or("main")
            ),
        ))
    }

    /// TODO-CORE-WIRE: `model_manager::load` (READY -> LOADED).
    pub async fn load_model(model_id: &str) -> Result<(), CoreStubError> {
        Err(not_wired(
            "load_model",
            format!("would load model_id={model_id} into its runtime."),
        ))
    }

    /// TODO-CORE-WIRE: `model_manager::unload` (LOADED -> READY).
    pub async fn unload_model(model_id: &str) -> Result<(), CoreStubError> {
        Err(not_wired(
            "unload_model",
            format!("would unload model_id={model_id} and free VRAM/RAM."),
        ))
    }

    /// TODO-CORE-WIRE: read `models` table via SQLx (docs/04 §4.1).
    pub async fn list_models() -> Result<Vec<ModelSummary>, CoreStubError> {
        Err(not_wired(
            "list_models",
            "would SELECT id,name,repository,status,capabilities FROM models.".into(),
        ))
    }

    /// TODO-CORE-WIRE: `hardware_manager::detect` (`HardwareBackend` trait,
    /// docs/07); 4GB profile is the reference low-end (AGENTS.md rule 6).
    pub async fn hardware_info() -> Result<HardwareSummary, CoreStubError> {
        Err(not_wired(
            "hardware_info",
            "would return CPU/RAM/GPU/VRAM/CUDA/driver/storage via HardwareBackend::detect.".into(),
        ))
    }

    /// TODO-CORE-WIRE: `runtime_manager` registry; MVP set is
    /// transformers/diffusers/llama.cpp (docs/03 §3.1).
    pub async fn list_runtimes() -> Result<Vec<RuntimeSummary>, CoreStubError> {
        Err(not_wired(
            "list_runtimes",
            "would list transformers/diffusers/llama.cpp with NOT_INSTALLED..RUNNING status.".into(),
        ))
    }

    /// TODO-CORE-WIRE: `scheduler` -> `RuntimeAdapter::prepare` -> `run`
    /// (AGENTS.md rule 3: UI calls `prepare->run` only). Must VRAM-check
    /// BEFORE load (AGENTS.md rule 6) and persist the generation row +
    /// output sidecar (docs/04 §4.1, §4.5) on success.
    pub async fn generate(req: &GenerateRequest) -> Result<GenerationResult, CoreStubError> {
        Err(not_wired(
            "generate",
            format!(
                "would run model={} prompt_len={} via scheduler + RuntimeAdapter.",
                req.model,
                req.prompt.len()
            ),
        ))
    }
}
