//! Nexora desktop shell (Tauri 2).
//!
//! One `#[tauri::command]` per entry in `CAPABILITY_NOTES.md`, each
//! delegating to the same `nexora` core fns as the REST layer (`src/api`).
//!
//! WIRING STATUS: `src-tauri` is a separate crate that must not gain a
//! `nexora` dependency without a `Cargo.toml` change (out of scope), so
//! every command that needs a manager currently returns `E_CORE_NOT_WIRED`
//! (`TODO-WIRE` cites the exact core fn + REST counterpart). The single
//! exception is the `trust_remote_code` gate inside `generation_prepare`,
//! which is REAL inline logic mirroring `nexora::security::trust::
//! gate_custom_code` — Unverified models return `E_CUSTOM_CODE` with the
//! View-Files / Sandbox / Cancel payload and never execute.
//!
//! Security: `trust_remote_code` models must go through the View Files /
//! Run in Sandbox / Cancel gate before any `prepare` call (AGENTS.md 5).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared error envelope: every failure carries { error_code } mapped in
// components/ErrorBanner.tsx (CUDA_OOM, VRAM_FIT, ...).
// ---------------------------------------------------------------------------

/// Structured backend error. Serialized to the frontend as
/// `{ error_code, message, fix }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandError {
    pub error_code: String,
    pub message: String,
    pub fix: String,
}

impl CommandError {
    /// Stub for commands whose core manager exists but is not wired to this
    /// command yet. Mirrors `core_stub::not_wired` (`E_CORE_NOT_WIRED`) used
    /// by the REST layer (`src/api/mod.rs`).
    pub fn core_not_wired(feature: &'static str, detail: impl Into<String>) -> Self {
        Self {
            error_code: "E_CORE_NOT_WIRED".into(),
            message: format!("TODO-CORE-WIRE: {feature} not wired to core yet. {}", detail.into()),
            fix: "Manager exists but is not wired to this endpoint yet (TODO-WIRE). Track docs/12-BUILD-TASKS.md Phase C.".into(),
        }
    }

    pub fn custom_code(repo: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            error_code: "E_CUSTOM_CODE".into(),
            message: format!("{} ships custom code ({}); consent required", repo.into(), detail.into()),
            fix: "Review the files via [View Files]. Only continue with [Run in Sandbox] if you trust the publisher; otherwise Cancel.".into(),
        }
    }
}

type Cmd<T> = Result<T, CommandError>;

// ---------------------------------------------------------------------------
// Hardware (docs/07)
// ---------------------------------------------------------------------------

/// TODO-WIRE: delegate to `nexora::hardware::{NvidiaBackend,CpuBackend}::
/// HardwareBackend::detect` — the same core fn `GET /hardware` will use
/// (`core_stub::hardware_info`). Returns the dashboard `HardwareInfo`.
#[tauri::command]
fn hardware_detect() -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "hardware_detect",
        "would return CPU/RAM/GPU/VRAM/CUDA/driver/storage via HardwareBackend::detect.",
    ))
}

// ---------------------------------------------------------------------------
// Models + HF (docs/05, docs/04 §4.2 state machine)
// ---------------------------------------------------------------------------

/// TODO-WIRE: `core_stub::parse_hf_url` (real, pure) + analyzer VALIDATING
/// step -> `model_manager::discover` (DISCOVERED -> ANALYZING -> SUPPORTED,
/// docs/04 §4.2). Same core path as `POST /models/install` validation.
#[tauri::command]
fn analyze_model(url: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "analyze_model",
        format!("would analyze url={url} via analyzer + model_manager::discover."),
    ))
}

/// TODO-WIRE: `nexora::hf::client` search (same fn a future `GET /hf/search`
/// REST endpoint would use). Returns `[{ id, likes? }]`.
#[tauri::command]
fn hf_search(query: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "hf_search",
        format!("would search Hub for query={query} via hf::client."),
    ))
}

/// TODO-WIRE: `storage::layout` file listing + sha256/dedup hashes
/// (docs/04 §4.5). Returns `[{ path, bytes, sha256, dedup_hash }]`.
#[tauri::command]
fn model_files(model_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "model_files",
        format!("would list files for model_id={model_id} via storage::layout."),
    ))
}

/// MUST memory-gate via the VRAM manager BEFORE load; returns
/// `{ ok: false, error_code: "VRAM_FIT" | "CUDA_OOM" | ... }` on refusal.
/// TODO-WIRE: `hardware::vram_fits` gate, then `model_manager::load`
/// (READY -> LOADED, docs/04 §4.2) — same as `POST /models/load`.
#[tauri::command]
fn model_load(model_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "model_load",
        format!("would VRAM-gate then load model_id={model_id} via model_manager::load."),
    ))
}

/// TODO-WIRE: `model_manager::unload` (LOADED -> READY) — same as
/// `POST /models/unload`.
#[tauri::command]
fn model_unload(model_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "model_unload",
        format!("would unload model_id={model_id} via model_manager::unload."),
    ))
}

// ---------------------------------------------------------------------------
// Downloads (docs/08). When wired, emit `download-progress` events:
// `{ modelId, bytesDone, bytesTotal, speedBps, etaSecs, state }`
// (stores/downloads.ts `applyProgress` consumes them).
// ---------------------------------------------------------------------------

/// TODO-WIRE: `download_manager` start + VRAM/disk precheck (AGENTS.md 6).
#[tauri::command]
fn download_start(model_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "download_start",
        format!("would start resumable download for model_id={model_id} via download_manager."),
    ))
}

/// TODO-WIRE: `download_manager` pause (keeps partial + checksum state).
#[tauri::command]
fn download_pause(model_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "download_pause",
        format!("would pause download for model_id={model_id}."),
    ))
}

/// TODO-WIRE: `download_manager` resume (range request + hash verify).
#[tauri::command]
fn download_resume(model_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "download_resume",
        format!("would resume download for model_id={model_id}."),
    ))
}

/// TODO-WIRE: `download_manager` cancel (drops partial files).
#[tauri::command]
fn download_cancel(model_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "download_cancel",
        format!("would cancel download for model_id={model_id}."),
    ))
}

// ---------------------------------------------------------------------------
// Runtimes (docs/02 §2.4, docs/04 §4.3 state machine)
// ---------------------------------------------------------------------------

/// TODO-WIRE: `runtime_manager` install into the isolated env (docs/09).
#[tauri::command]
fn runtime_install(runtime_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "runtime_install",
        format!("would install runtime_id={runtime_id} via runtime_manager."),
    ))
}

/// TODO-WIRE: `runtime_manager` spawn supervised child process.
#[tauri::command]
fn runtime_start(runtime_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "runtime_start",
        format!("would start runtime_id={runtime_id} as a supervised child process."),
    ))
}

/// TODO-WIRE: `runtime_manager` stop child process. Returns `-> { healthy }`.
#[tauri::command]
fn runtime_stop(runtime_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "runtime_stop",
        format!("would stop runtime_id={runtime_id}."),
    ))
}

/// TODO-WIRE: `RuntimeAdapter::health_check` — same core fn the REST
/// `GET /runtimes` health field will use. Returns `-> { healthy }`.
#[tauri::command]
fn runtime_health(runtime_id: String) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "runtime_health",
        format!("would health-check runtime_id={runtime_id} via RuntimeAdapter::health_check."),
    ))
}

// ---------------------------------------------------------------------------
// Generation — UI calls prepare -> run ONLY, never the runtime directly
// (AGENTS.md rule 3; RuntimeAdapter contract docs/06).
// ---------------------------------------------------------------------------

/// Response shape the UI consumes: `{ ok: true } | { ok: false,
/// error_code, ... }` (stores/generation.ts branches on `data.ok`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareResponse {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    /// Trust-gate choices surfaced as View Files / Run in Sandbox / Cancel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actions: Option<Vec<String>>,
}

/// Trust-gate actions offered on `E_CUSTOM_CODE` (TRUST_BLOCK).
const TRUST_ACTIONS: [&str; 3] = ["view-files", "sandbox", "cancel"];

fn trust_denied(repo: &str, detail: &str) -> PrepareResponse {
    let e = CommandError::custom_code(repo, detail);
    PrepareResponse {
        ok: false,
        error_code: Some(e.error_code),
        message: Some(e.message),
        fix: Some(e.fix),
        actions: Some(TRUST_ACTIONS.iter().map(|s| s.to_string()).collect()),
    }
}

/// REAL trust gate (mirrors `nexora::security::trust::gate_custom_code`):
/// `Blocked` repos always fail; `Unverified` / `trust_remote_code` /
/// custom-`.py` models require explicit `user_decision = "sandbox"`
/// ([Run in Sandbox]); `view-files`, `cancel`, or no decision returns
/// `E_CUSTOM_CODE` with the View-Files/Sandbox/Cancel payload and NEVER
/// executes. After consent, preparation itself is still TODO-WIRE
/// (`scheduler` -> `RuntimeAdapter::prepare`, AGENTS.md 3) and returns
/// `E_CORE_NOT_WIRED` — the gate passing never implies execution.
#[tauri::command]
fn generation_prepare(
    model_id: String,
    capability: String,
    params: serde_json::Value,
    trust_level: Option<String>,
    trust_remote_code: Option<bool>,
    has_custom_py: Option<bool>,
    user_decision: Option<String>,
) -> Cmd<PrepareResponse> {
    let trust = trust_level.as_deref().unwrap_or("unknown");
    let custom = trust_remote_code.unwrap_or(false) || has_custom_py.unwrap_or(false);

    // Blocked repos always fail, even with consent.
    if trust.eq_ignore_ascii_case("blocked") {
        return Ok(trust_denied(&model_id, "blocked: dangerous or incompatible content"));
    }
    // Unverified trust (or any custom executable code) needs the gate.
    if trust.eq_ignore_ascii_case("unverified") || trust.eq_ignore_ascii_case("untrusted") || custom {
        match user_decision.as_deref() {
            Some("sandbox") => {} // explicit [Run in Sandbox] consent: gate opens
            Some("view-files") => {
                return Ok(trust_denied(
                    &model_id,
                    "review requested — confirm [Run in Sandbox] or [Cancel] afterwards",
                ))
            }
            _ => {
                return Ok(trust_denied(
                    &model_id,
                    "custom repo code requires explicit [Run in Sandbox] consent",
                ))
            }
        }
    }

    // Gate passed — preparation itself is not wired yet.
    let _ = (capability, params);
    Err(CommandError::core_not_wired(
        "generation_prepare",
        format!("would prepare model_id={model_id} via scheduler + RuntimeAdapter::prepare."),
    ))
}

/// TODO-WIRE: `scheduler` -> `RuntimeAdapter::run` (AGENTS.md 3), persisting
/// the generation row + output sidecar (docs/04 §4.1, §4.5) — same core fn
/// as `POST /generate` (`core_stub::generate`). Every failure carries
/// `{ error_code }`.
#[tauri::command]
fn generation_run(
    model_id: String,
    capability: String,
    params: serde_json::Value,
) -> Cmd<serde_json::Value> {
    let _ = (capability, params);
    Err(CommandError::core_not_wired(
        "generation_run",
        format!("would run model_id={model_id} via scheduler + RuntimeAdapter::run."),
    ))
}

// ---------------------------------------------------------------------------
// Environments (docs/09) — isolated envs, never global pip; pin + rollback.
// ---------------------------------------------------------------------------

/// TODO-WIRE: `env::manager` list (1-env-per-runtime-kind, docs/09 §9.1).
#[tauri::command]
fn env_list() -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "env_list",
        "would list isolated envs via env::manager.",
    ))
}

/// TODO-WIRE: `EnvPin::compatible_with` reuse check (superset envs reusable).
#[tauri::command]
fn env_reuse_check() -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "env_reuse_check",
        "would check EnvPin compatibility via env::manager.",
    ))
}

/// TODO-WIRE: `env::manager` rollback to previous pins.
#[tauri::command]
fn env_rollback() -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "env_rollback",
        "would roll back env pins via env::manager.",
    ))
}

// ---------------------------------------------------------------------------
// Workflows / jobs (docs/10)
// ---------------------------------------------------------------------------

/// TODO-WIRE: `scheduler::queue` + `jobs` graph execution.
#[tauri::command]
fn workflow_run(graph: serde_json::Value) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "workflow_run",
        format!("would execute workflow graph via scheduler::queue (graph={graph})."),
    ))
}

// ---------------------------------------------------------------------------
// Logs (docs/11 §11.6, scopes Runtime/Download/Model/System). When wired,
// also emit `log-line` events for LogViewer live tail.
// ---------------------------------------------------------------------------

/// TODO-WIRE: `storage`/`tracing` log store query. Returns `LogLine[]`.
#[tauri::command]
fn logs_query(scope: Option<String>, limit: Option<u32>) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "logs_query",
        format!(
            "would query logs scope={} limit={} via the log store.",
            scope.as_deref().unwrap_or("All"),
            limit.unwrap_or(200),
        ),
    ))
}

// ---------------------------------------------------------------------------
// Settings (persisted via settings table, docs/04 §4.1)
// ---------------------------------------------------------------------------

/// TODO-WIRE: `SELECT key,value FROM settings` via SQLx.
/// Returns `Record<string,string>`.
#[tauri::command]
fn settings_get_all() -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "settings_get_all",
        "would SELECT key,value FROM settings.",
    ))
}

/// TODO-WIRE: upsert into `settings` table via SQLx.
#[tauri::command]
fn settings_set(key: String, value: serde_json::Value) -> Cmd<serde_json::Value> {
    Err(CommandError::core_not_wired(
        "settings_set",
        format!("would upsert settings key={key} value={value}."),
    ))
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            hardware_detect,
            analyze_model,
            hf_search,
            model_files,
            model_load,
            model_unload,
            download_start,
            download_pause,
            download_resume,
            download_cancel,
            runtime_install,
            runtime_start,
            runtime_stop,
            runtime_health,
            generation_prepare,
            generation_run,
            env_list,
            env_reuse_check,
            env_rollback,
            workflow_run,
            logs_query,
            settings_get_all,
            settings_set
        ])
        .run(tauri::generate_context!())
        .expect("error while running Nexora");
}
