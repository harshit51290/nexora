# Nexora — Docs Index (AI Builder Entry Point)

> Read this file first. It maps the full product plan (104 sections) into buildable docs with zero missing coverage.

## Product in one sentence
> **A universal local AI runtime manager that automatically discovers the appropriate inference backend for AI models.** Not "an app that runs every HF model".

Core loop the product must answer:
```
WHAT IS THIS MODEL? -> WHAT CAN RUN IT? -> WHAT DOES MY COMPUTER SUPPORT?
-> WHICH RUNTIME IS BEST? -> HOW DO I INSTALL IT? -> HOW DO I CONFIGURE IT? -> HOW DO I RUN IT?
```

## Architecture in one diagram
```
React(TS+Tailwind) -> Tauri2 -> Rust Core (Model/Runtime/Download/Hardware/Scheduler/Security/Plugin/API)
-> Model Analyzer + Environment + Job Scheduler
-> Transformers | Diffusers | llama.cpp | ONNX | Audio | ComfyUI (+ plugins)
-> Hardware Backend (CUDA/DirectML/CPU) -> NVIDIA GPU
```
Rule: **Rust = orchestration. Existing runtimes = execution.** Never build own NN engine/CUDA/diffusion/tensor lib (see `01-VISION-PRD.md` § What-NOT-to-build).

## Doc map (all 104 source sections covered)

| Doc | Covers |
|-----|--------|
| `01-VISION-PRD.md` | vision, problem Model-vs-Runtime, principle, platforms, naming, business, risks, NOT-build, long-term vision, core rule |
| `02-ARCHITECTURE.md` | stack, folder layout, layers, core modules, perf/GPU/HW abstraction, Rust value vs Python necessity |
| `03-ROADMAP-MILESTONES.md` | MVP scope+flow, v0.2-1.0, 15 milestones, dev order, 12-month roadmap, first concrete goal |
| `04-DATA-MODEL.md` | SQLite schema, model/runtime state machines, capability system, storage layout |
| `05-HF-INTEGRATION.md` | HF import, analyzer, 5-category classification, adapter registry, unsupported/experimental, auto-py-detect, HF search/login, licensing |
| `06-RUNTIME-ADAPTERS.md` | `RuntimeAdapter` trait, Transformers/Diffusers/llama.cpp/ONNX/Audio/Video/ComfyUI, plugin system |
| `07-HARDWARE.md` | detection, VRAM manager, profiles, compat score, perf estimate, quantization, auto-optimization, recommendations, concurrent/resource mgmt, background loading |
| `08-DOWNLOAD-STORAGE.md` | install flow, download manager, storage manager, dedup, cache layout, outputs, history |
| `09-ENV-SECURITY.md` | env manager/reuse, container defer, version locking, rollback, updates, security/trust levels/isolation |
| `10-API-CLI.md` | unified REST, OpenAI-compat, local API table, CLI, Web UI, Discord (future), job queue/batch, workflows/templates, scheduler |
| `11-UI-UX.md` | nav sections, Home, library, model cards, unified generation UI (text/image/audio), beginner/advanced/dev modes, logs + error translation |
| `12-BUILD-TASKS.md` | phased autonomous tasks with acceptance criteria (maps to milestones) |
| `13-FRONTEND-RS.md` | pure-Rust UI track (Dioxus 0.7): Context7 decision record, parity plan |
| `AGENTS.md` (repo root) | hard rules for AI builders |

## Build order (do not skip)
```
Foundation -> Hardware -> HF API -> Downloader -> Analyzer -> Transformers -> Diffusers -> GGUF
-> Unified UI -> Audio -> ComfyUI -> Video -> Plugins -> Experimental
```
Full MVP = HF URL -> Analyze -> Install -> Configure -> Run -> Generate (Text + Image, NVIDIA/CPU only).

## Target matrix
- Primary: **Windows** (NVIDIA gamers/laptops/iGPU). Later Linux, then macOS. Tauri for cross-platform UI; platform-specific runtime logic in backend.
- GPU start: NVIDIA CUDA. Then DirectML/ONNX/AMD/Intel/Apple via `GpuBackend`/`HardwareBackend` traits. Never assume NVIDIA forever.
