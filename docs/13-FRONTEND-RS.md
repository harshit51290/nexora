# 13 — Pure-Rust UI Track (Dioxus), via Context7

## 13.1 Decision (2026-10-09, Context7 evidence)
| Candidate | Verdict | Reason |
|---|---|---|
| **Dioxus 0.7** (`/dioxuslabs/dioxus`) | **CHOSEN** | Desktop renderer is built **off Tauri** (wry/tao) — keeps `tauri.conf.json`, capabilities, nsis/msi bundling, and every backend contract. RSX maps 1:1 from the React components; signals replace Zustand. Entry per official readme: `dioxus_desktop::launch(app)` with `dioxus` + `dioxus-desktop` deps. |
| iced 0.14 (`/iced-rs/iced`) | Rejected for now | Best pure-native architecture (Elm + Task/subscriptions fit downloads/generation), but drops Tauri entirely — new bundling story, IPC/commands/capabilities reimplemented, theming from scratch. Revisit if the webview ever goes. |
| egui/eframe (`/emilk/egui`) | Rejected | Immediate-mode + fastest prototype, but a retained 10-page dashboard with forms is awkward, theming weakest, least "polished native app" (docs/11 bar). |

## 13.2 What `frontend-rs/` contains
`nexora-ui` bin: `main` (launch) → `app::App` (owns all signals) → `pages` (10 sections) + `components` (sidebar, analyze bar, hardware card, model cards, compat bars, capability-driven generate form, error banner, log viewer) → `backend` (direct core calls) → `state` (Section/Mode/seeds).
- **Live today (no IPC):** hardware detect (`NvidiaBackend::detect`), URL parse + pre-classifier + REAL hardware-aware quant/dtype/offload recommendation (`recommend_config` on live `MemoryInfo`) and REAL compat bars (`compat_score`) — size unknown until Hub fetch, labeled pre-download on the card.
- **Stubbed (TODO-WIRE-UI):** install/generate — same coded-error contract as the API; `uar`/API path works now.
- `#[component]` used once (paramless `App`); all other render units are plain fns — no Props-derive risk.

## 13.3 Status: Dioxus is the UI, React removed
`frontend/` (React/TS) was deleted; `frontend-rs/` (`nexora-ui`) is the
desktop UI. Primary dev loop is `cargo run -p nexora-ui` (needs the MSVC
linker like everything else). `src-tauri/` is suspended: `tauri.conf.json`
no longer has npm hooks and `frontendDist` points at the future Dioxus web
dist (`../frontend-rs/dist`) — `tauri dev/bundle` resumes when the Dioxus
web target lands. The 23 Tauri commands stay as the web-target IPC contract.
`cargo fetch` locks 785 crates; full `cargo check` waits on the MSVC linker.
