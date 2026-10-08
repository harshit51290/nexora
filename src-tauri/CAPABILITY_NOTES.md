# Nexora — Tauri wiring notes (for the backend agent)

The frontend (React + TS + Tailwind + Zustand, `frontend/`) is scaffolded and
calls the commands below via `invoke`. **None are implemented yet** — the
backend agent owns `src-tauri/src/main.rs`, `build.rs`, `icons/`, and the
`#[tauri::command]` implementations.

## Expected invoke names (frontend calls these and only these)

```js
// Hardware (docs/07)
// - hardware_detect() -> HardwareInfo { cpu, ramGb, gpu, vramGb,
//     cudaAvailable, cudaVersion?, driverVersion?, storageFreeGb, os, arch, status }
//
// Models + HF (docs/05, docs/04 §4.2 state machine)
// - analyze_model(url) -> ModelEntry (DISCOVERED -> ANALYZING -> SUPPORTED)
// - hf_search(query) -> [{ id, likes? }]
// - model_files(model_id) -> [{ path, bytes, sha256, dedup_hash }]
// - model_load(model_id) / model_unload(model_id)
//   // MUST memory-gate via the VRAM manager BEFORE load; return
//   // { ok: false, error_code: "VRAM_FIT" | "CUDA_OOM" | ... } on refusal.
//
// Downloads (docs/08)
// - download_start(model_id) / download_pause(model_id)
// - download_resume(model_id) / download_cancel(model_id)
// // + emit `download-progress` events:
// //   { modelId, bytesDone, bytesTotal, speedBps, etaSecs, state }
// // stores/downloads.ts `applyProgress` consumes them.
//
// Runtimes (docs/02 §2.4, docs/04 §4.3 state machine)
// - runtime_install(runtime_id) / runtime_start(runtime_id)
// - runtime_stop(runtime_id) / runtime_health(runtime_id) -> { healthy }
//
// Generation — UI calls prepare -> run ONLY, never the runtime directly
// (AGENTS.md rule 3; RuntimeAdapter contract docs/06).
// - generation_prepare({ model_id, capability, params })
//     -> { ok: true } | { ok: false, error_code }
// - generation_run({ model_id, capability, params }) -> GenerationRecord
//   // Every failure carries { error_code } mapped in
//   // components/ErrorBanner.tsx (CUDA_OOM, VRAM_FIT, ...).
//
// Environments (docs/09) — isolated envs, never global pip; pin + rollback.
// - env_list() / env_reuse_check() / env_rollback()
//
// Workflows / jobs (docs/10)
// - workflow_run(graph)
//
// Logs (docs/11 §11.6, scopes Runtime/Download/Model/System)
// - logs_query({ scope, limit }) -> LogLine[]
// // + emit `log-line` events for LogViewer live tail.
//
// Settings (persisted via settings table, docs/04 §4.1)
// - settings_get_all() -> Record<string,string>
// - settings_set({ key, value })
```

## Backend-agent TODO (not frontend scope)

1. `src-tauri/src/main.rs` + `build.rs` (`tauri::Builder`, register commands above).
2. `src-tauri/icons/` (32x32, 128x128, 128x128@2x, icon.icns, icon.ico) — required by the `nsis`/`msi` bundle targets.
3. Capabilityidentifier/permissions hardening in `capabilities/` as commands land.
4. Trust gate: `trust_remote_code` / custom `.py` must surface a View / Sandbox / Cancel prompt (TRUST_BLOCK) — never auto-trust.
5. Compat + VRAM check BEFORE download/load (4GB GTX 1050 profile is the reference low-end).
