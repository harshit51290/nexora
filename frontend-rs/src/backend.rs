//! Direct calls into the `nexora` core (same fns as REST/CLI).
//! Only synchronous, side-effect-free core APIs are used here:
//! hardware snapshot + URL parse + pre-classifier. Anything needing the
//! DB, downloads, or child processes returns a coded error pointing at
//! the `uar` CLI / API until the async service handle is wired
//! (TODO-WIRE-UI, mirrors `src-tauri/CAPABILITY_NOTES.md` NEED table).

use nexora::api::core_stub;
use nexora::hardware::{HardwareBackend, HardwareInfo, NvidiaBackend};

use crate::state::Analyzed;

/// Real machine snapshot (CPU baseline + `nvidia-smi` when present).
pub fn detect_hardware() -> HardwareInfo {
    NvidiaBackend::detect()
}

/// Real parse + real pre-classifier (both pinned by `tests/mvp_flow.rs`).
/// Full registry/VRAM scoring lands with the async service handle.
pub fn analyze_url(url: &str) -> Result<Analyzed, String> {
    let id = core_stub::parse_hf_url(url).map_err(|e| format!("[{}] {}", e.code, e.message))?;
    let rec = core_stub::classify_model(&id);
    Ok(Analyzed {
        id,
        task: rec.task,
        runtime: rec.runtime,
        notes: rec.notes,
    })
}

/// Install / generate need the async `CoreHandle` (DB + scheduler +
/// runtimes). Coded stub until TODO-WIRE-UI — same contract as the API.
pub fn install_not_wired() -> String {
    "[E_CORE_NOT_WIRED] Install from this UI needs the async service handle. Use `uar install <owner/model>` for now.".into()
}

pub fn generate_not_wired() -> String {
    "[E_CORE_NOT_WIRED] Generate from this UI needs the scheduler + runtimes. Use `uar run` / the API for now.".into()
}
