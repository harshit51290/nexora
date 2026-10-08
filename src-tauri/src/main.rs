//! Nexora desktop shell (Tauri 2).
//!
//! TODO-WIRE: register one `#[tauri::command]` per entry in
//! `CAPABILITY_NOTES.md` (`hardware_detect`, `analyze_model`, `model_load`,
//! `download_start`, `generation_prepare`/`generation_run`, …), each
//! delegating to the same `nexora` core fns as the REST/CLI layers.
//! Security: `trust_remote_code` models must go through the View Files /
//! Run in Sandbox / Cancel gate before any `prepare` call.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("error while running Nexora");
}
