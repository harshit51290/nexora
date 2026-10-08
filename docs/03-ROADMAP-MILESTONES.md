# 03 — Scope, Milestones, Roadmap

## 3.1 MVP v0.1 (extremely specific — do not expand)
- Input: HF URL. Auto-detect: architecture, task, format, runtime, estimated reqs.
- Runtimes: **Transformers, Diffusers, llama.cpp only**. Tasks: **text + image generation only**. HW: **NVIDIA CUDA + CPU fallback**.
- App: Dioxus 0.7 (pure Rust), Rust backend, SQLite.
- Flow: HF URL -> Analyze -> Install -> Configure -> Run -> Generate.
- Victory 1: `START -> paste SD URL -> "Stable Diffusion/Diffusers/4GB low-VRAM cfg" -> Install -> Download -> runtime install -> Validate -> Prompt -> IMAGE`.
- Victory 2: paste Qwen GGUF -> llama.cpp + Q4 rec -> Install -> Chat.
- Victory 3 (proves universal arch; may slip to v0.3 if needed): paste TTS -> audio runtime -> Text -> Audio.

## 3.2 Version plan
- v0.2: LLM+GGUF polish, llama.cpp, chat UI, streaming, model library (Image+Text done).
- v0.3: TTS + Whisper + audio playback (Text+Image+Audio).
- v0.4: ComfyUI + workflows + image-to-video/video-gen.
- v0.5: ONNX + DirectML + AMD/Intel/Apple (demand-driven).
- v1.0: universal discovery, auto runtime selection, HW optimization, mgmt, Text/Image/Audio/Video/Vision, API+CLI+workflows+plugins.

## 3.3 Dev order (biggest mistake = "support everything first")
`Rust foundation -> Hardware -> HF API -> Downloader -> Analyzer -> Transformers -> Diffusers -> GGUF -> Unified UI -> Audio -> ComfyUI -> Video -> Plugins -> Experimental`.

## 3.4 Milestones M1–M15 (goal = done-criteria)
1. Rust/Tauri foundation (app launches, SQLite, logging, settings).
2. Hardware detection (CPU/RAM/GPU/VRAM/OS/storage dashboard).
3. HF integration (URL -> metadata/files).
4. Download manager (resume/pause/cancel/progress/checksum/cache; large models reliable).
5. Model analyzer (config/model_index/file/arch/task detection; auto-classify).
6. Transformers adapter (text-gen/embedding/classification/vision; first HF run).
7. Diffusers adapter (SD/SDXL image-gen).
8. llama.cpp (GGUF, quantized LLMs efficient).
9. Unified UI (chat, image-gen, config, gallery; user never sees runtime name unless advanced).
10. Runtime manager (env creation, deps, versions; auto-setup).
11. Low-VRAM (detection, offload, precision/quant, monitoring; 4GB GPUs work).
12. Audio (Whisper, TTS, playback).
13. ComfyUI (workflow exec, custom nodes; advanced image/video).
14. API (REST + OpenAI-compat + WS streaming; external apps can use runner).
15. Plugins (third-party runtimes).

## 3.5 12-month roadmap
M1 foundation | M2 hardware | M3 HF | M4 downloads | M5 transformers | M6 diffusers | M7 GGUF | M8 unified UI | M9 audio | M10 ComfyUI | M11 API+plugins+CLI | M12 polish (security, installer, auto-update, crash recovery, docs, tests, release).
