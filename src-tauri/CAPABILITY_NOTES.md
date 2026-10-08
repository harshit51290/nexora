# Nexora — Tauri wiring notes (for the backend agent)

The frontend (React + TS + Tailwind + Zustand, `frontend/`) calls the
commands below via `invoke` (typed wrappers in
`frontend/src/lib/tauri.ts`). **All 23 are registered** in
`src-tauri/src/main.rs` via `tauri::generate_handler!`.

## Wiring status (2026-10-08)

- REGISTERED + REAL LOGIC (1): `generation_prepare` enforces the
  `trust_remote_code` gate inline (mirrors
  `nexora::security::trust::gate_custom_code`): `Blocked` always fails;
  `Unverified`/`trust_remote_code`/custom-`.py` without
  `user_decision = "sandbox"` returns `{ ok: false, error_code:
  "E_CUSTOM_CODE", actions: ["view-files", "sandbox", "cancel"] }`
  (TRUST_BLOCK) and never executes. Post-consent preparation is still
  `E_CORE_NOT_WIRED` — the gate passing never implies execution.
- REGISTERED + `E_CORE_NOT_WIRED` stub (22, all NEED): everything else.
  Each stub cites the exact core fn + REST counterpart in its `TODO-WIRE`.
  `src-tauri` cannot link the `nexora` crate without a `Cargo.toml` change
  (out of scope), so delegation is blocked on adding that dependency and
  replacing each stub body — see NEED list below.
- `capabilities/default.json` intentionally lists only `core:default`:
  app-owned `#[tauri::command]`s are allow-by-default in Tauri v2
  capabilities (the `permissions` array gates plugin/core APIs, not own
  commands), so there is no valid permission identifier to add per command.

## NEED list (stub -> real core fn)

| Command | Real core fn (REST layer uses the same) |
|---|---|
| `hardware_detect` | `hardware::HardwareBackend::detect` (`GET /hardware`) |
| `analyze_model` | `core_stub::parse_hf_url` (real) + analyzer + `model_manager::discover` |
| `hf_search` | `hf::client` search |
| `model_files` | `storage::layout` listing |
| `model_load` | `hardware::vram_fits` gate + `model_manager::load` (`POST /models/load`) |
| `model_unload` | `model_manager::unload` (`POST /models/unload`) |
| `download_start/pause/resume/cancel` | `download::manager` (+ emit `download-progress`) |
| `runtime_install/start/stop/health` | `runtime_manager` / `RuntimeAdapter::health_check` |
| `generation_run` | `scheduler` -> `RuntimeAdapter::run` (`POST /generate`) |
| `env_list/reuse_check/rollback` | `env::manager` / `EnvPin::compatible_with` |
| `workflow_run` | `scheduler::queue` + `jobs` |
| `logs_query` | log store (+ emit `log-line`) |
| `settings_get_all/set` | SQLx `settings` table |

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
