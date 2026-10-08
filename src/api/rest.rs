//! Local REST surface (`docs/10-API-CLI.md` §10.1).
//!
//! `GET /models /hardware /runtimes | POST /models/install /models/load
//! /models/unload /generate`, plus `POST /v1/generate` and
//! `POST /models/downloads/{pause,resume,cancel}`. The runner picks the
//! backend. Every handler terminates at the real core (`service`); failures
//! keep the `ErrorBody {code,message,fix}` shape. The HF-URL validation in
//! `install_model` rejects bad refs with 400 before any manager runs.

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};

use super::{
    core_stub, ApiError, AppState, EstimateRequest, GenerateRequest, GenerationResult,
    HardwareSummary, InstallRequest, InstallResponse, LoadRequest, ModelSummary, RuntimeSummary,
};

/// Routes mounted at `/` (see [`super::build_router`]).
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/models", get(list_models))
        .route("/hardware", get(get_hardware))
        .route("/runtimes", get(list_runtimes))
        .route("/models/install", post(install_model))
        .route("/models/load", post(load_model))
        .route("/models/unload", post(unload_model))
        .route("/generate", post(generate))
        .route("/v1/generate", post(generate))
        .route("/models/downloads/pause", post(downloads_pause))
        .route("/models/downloads/resume", post(downloads_resume))
        .route("/models/downloads/cancel", post(downloads_cancel))
        .route("/models/estimate", post(estimate_model))
        .with_state(state)
}

async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<ModelSummary>>, ApiError> {
    core_stub::list_models()
        .await
        .map(Json)
        .map_err(|e| ApiError::from_core(state.version, e))
}

async fn get_hardware(
    State(state): State<AppState>,
) -> Result<Json<HardwareSummary>, ApiError> {
    core_stub::hardware_info()
        .await
        .map(Json)
        .map_err(|e| ApiError::from_core(state.version, e))
}

async fn list_runtimes(
    State(state): State<AppState>,
) -> Result<Json<Vec<RuntimeSummary>>, ApiError> {
    core_stub::list_runtimes()
        .await
        .map(Json)
        .map_err(|e| ApiError::from_core(state.version, e))
}

async fn install_model(
    State(state): State<AppState>,
    Json(req): Json<InstallRequest>,
) -> Result<Json<InstallResponse>, ApiError> {
    // Real validation that survives core wiring: reject bad refs with 400
    // before ever touching the download manager.
    let repo = core_stub::parse_hf_url(&req.repo).map_err(|e| {
        ApiError::bad_request(
            state.version,
            e.code,
            e.message,
            "Pass a repo id (owner/model) or URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5.",
        )
    })?;
    // ModelManager install + DownloadManager fetch (docs/08 flow).
    let model_id = core_stub::install_model(&repo, req.revision.as_deref())
        .await
        .map_err(|e| ApiError::from_core(state.version, e))?;
    Ok(Json(InstallResponse {
        model_id,
        status: "DOWNLOADING".to_string(),
    }))
}

async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // ModelManager load (READY -> LOADED, docs/04 §4.2).
    core_stub::load_model(&req.model_id)
        .await
        .map_err(|e| ApiError::from_core(state.version, e))?;
    Ok(Json(serde_json::json!({"model_id": req.model_id, "status": "LOADED"})))
}

async fn unload_model(
    State(state): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // ModelManager unload (LOADED -> READY).
    core_stub::unload_model(&req.model_id)
        .await
        .map_err(|e| ApiError::from_core(state.version, e))?;
    Ok(Json(serde_json::json!({"model_id": req.model_id, "status": "READY"})))
}

async fn estimate_model(
    State(state): State<AppState>,
    Json(req): Json<EstimateRequest>,
) -> Result<Json<crate::mem::MemEstimate>, ApiError> {
    // Weight/KV bytes WITHOUT downloading (hf-mem method, `src/mem`).
    core_stub::estimate(&req)
        .await
        .map(Json)
        .map_err(|e| ApiError::from_core(state.version, e))
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerationResult>, ApiError> {
    if req.prompt.trim().is_empty() {
        return Err(ApiError::bad_request(
            state.version,
            "E_EMPTY_PROMPT",
            "Prompt is empty.".to_string(),
            "Provide a non-empty prompt string.",
        ));
    }
    // Scheduler -> RuntimeAdapter prepare->run (AGENTS.md 3).
    core_stub::generate(&req)
        .await
        .map(Json)
        .map_err(|e| ApiError::from_core(state.version, e))
}

async fn downloads_pause(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    download_control(&state, "pause").await
}

async fn downloads_resume(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    download_control(&state, "resume").await
}

async fn downloads_cancel(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    download_control(&state, "cancel").await
}

async fn download_control(
    state: &AppState,
    action: &str,
) -> Result<Json<serde_json::Value>, ApiError> {
    let status = core_stub::download_control(action)
        .await
        .map_err(|e| ApiError::from_core(state.version, e))?;
    Ok(Json(serde_json::json!({"action": action, "state": status})))
}
