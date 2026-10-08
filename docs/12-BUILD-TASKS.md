# 12 — Autonomous Build Tasks (agent work queue)

> Execute in order. Each task lists done-criteria. Do not start next phase until current gates pass.

## Phase A — Foundation (M1–M2)
- A1 Tauri2+React+TS+Tailwind shell with 10 nav routes, Zustand stores, SQLite via SQLx + migrations, tracing+settings. Gate: app launches, empty pages render.
- A2 Hardware manager (`HardwareBackend` trait + NVIDIA/CPU impls): CPU/RAM/GPU/VRAM/CUDA/driver/storage dashboard. Gate: GTX1050-class machine shows correct VRAM.

## Phase B — HF + Downloads + Analyzer (M3–M5)
- B1 HF client: URL parse, Hub API, file list, token via OS store. Gate: paste URL -> metadata/files/license screen.
- B2 Download manager: resume/pause/cancel/retry/parallel/checksum/space-check/speed+ETA. Gate: 10GB+ model survives cancel+resume with valid hash.
- B3 Analyzer + registry: config/model_index/file detection, 5-category classify, `registry.json`, compat score + rec config. Gate: SD1.5->Diffusers, Qwen-GGUF->llama.cpp, unknown arch->unsupported card with Experimental offer.

## Phase C — MVP runtimes + UI (M6–M11)
- C1 Transformers adapter (isolated env, text-gen). Gate: small LLM generates text via `prepare->run`.
- C2 Diffusers adapter (SD1.5, FP16/offload/slicing/tiling). Gate: 512px image on 4GB.
- C3 llama.cpp adapter (GGUF Q4 default, offload layers). Gate: Qwen chat streams.
- C4 Unified Generate + library + history + outputs sidecars. Gate: user never needs runtime name in Beginner mode.
- C5 VRAM manager + perf estimates + auto-optimization. Gate: 70B on 16GB warns Poor before download; 4GB runs pass.

## Phase D — Audio/Video/API/Plugins (M12–M15)
- D1 Audio: Whisper STT + TTS + playback. D2 ComfyUI manager (install/nodes/workflow JSON/serve/execute). D3 REST + OpenAI-compat + WS + CLI (`install/run/models/hardware/generate/serve`). D4 Plugin manifest + sandbox exec. D5 Polish: installer, auto-update, crash recovery, error catalog, security audit, docs.

## Global gates
Rust orchestration only; isolated envs; child-process runtimes; no silent custom-code exec; version pinning + rollback; relocatable data dir; every feature needs ERROR-code + human fix + log scope.
