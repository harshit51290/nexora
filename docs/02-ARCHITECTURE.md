# 02 — Architecture & Tech Stack

## 2.1 Stack (locked for v0.1)
- Backend: **Rust** + Tokio, Reqwest, Serde, SQLx, SQLite, Sysinfo, Tauri 2, tracing, anyhow, thiserror, `tokio::process`.
- Desktop: **Dioxus 0.7 desktop UI in pure Rust** (`frontend-rs/nexora-ui`; Tauri-based renderer, no JS). Must feel native, not a web dashboard.
- State: Dioxus signals (`section/mode/models/hardware/analysis/status` in `frontend-rs`, owned by `App`).
- Perf rule: Rust = UI coordination/downloads/process mgmt/HW monitoring/scheduling. Python = ML inference. Native runtimes = optimized inference. Rust never processes tensors unless necessary.

## 2.2 Layering (no layer skipping)
```
UI -> Application -> Services -> Core -> Adapters -> External Runtime
```

## 2.3 Repo layout (create exactly)
```
nexora/
  src/
    core/ model/ runtime/ hardware/ download/ scheduler/ security/ api/ storage/ plugins/
    main.rs lib.rs
  runtimes/ transformers/ diffusers/ llama_cpp/ onnx/ comfyui/ audio/
  frontend-rs/ src/{main,app,backend,components,pages,state}.rs (Dioxus 0.7 UI)
  plugins/
  migrations/
  docs/
  AGENTS.md README.md
```
Data dir: `UniversalRunner/{models,runtimes,environments,cache,outputs,logs,plugins}/`; root relocatable (e.g. `D:\AI\Models`). See `04-DATA-MODEL.md`.

## 2.4 Core Rust modules (responsibilities)
- `model_manager`: install/remove/load/unload/metadata.
- `runtime_manager`: install/start/stop/health-check runtimes (child processes; app survives runtime crash).
- `hardware_manager`: CPU/RAM/GPU/VRAM/CUDA/driver/storage/OS/arch.
- `download_manager`: download/resume/checksum/cache (see `08-DOWNLOAD-STORAGE.md`).
- `scheduler`: jobs, GPU allocation, concurrency, queue.
- `security_manager`: permissions, sandbox, custom-code warnings.

## 2.5 Hardware abstraction (no NVIDIA hardcode)
```rust
trait HardwareBackend { fn detect() -> HardwareInfo; fn memory() -> MemoryInfo; fn capabilities() -> Capabilities; }
// impls: NVIDIABackend, AMDBackend, IntelBackend, AppleBackend, CPUBackend
trait GpuBackend { /* cuda / directml / cpu providers */ }
```
Start CUDA; add DirectML/ONNX/AMD/Intel/Metal behind trait.

## 2.6 Why Rust / where Python stays
Rust: reliability (long-running processes), perf (downloads/files), memory safety (huge files), concurrency (downloads/GPU/logs/jobs), single-binary distribution. Python remains: easiest path for research models; always launched as isolated `python.exe` inside envs. Rust is "the OS for AI runtimes".

## 2.7 Final target topology
```
Tauri UI -> Rust Core (Model/Runtime/Download/HW/Scheduler/Security/Plugin/API)
 -> Analyzer + Environment + Job Scheduler
 -> Transformers/PyTorch | Diffusers/ComfyUI | llama.cpp/GGUF
 -> Hardware Backend (CUDA/DirectML/CPU) -> GPU
```
