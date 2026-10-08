# 11 — UI/UX Spec

## 11.1 Nav (exact sections)
Home | Discover | Models | Workflows | Generate | Downloads | Environments | Hardware | Runtime | Settings.

## 11.2 Home (extremely simple)
Top: `Run AI models locally. Paste HF URL [url input] [Analyze Model]`. Mid: Hardware card (GPU GTX1050 4GB / RAM 16GB / CPU i7 / Status Ready). Bottom: Recently Used (Qwen, SD, Whisper, Kokoro).

## 11.3 Models library + cards
Cards: name, task, params (7B), VRAM, runtime, `[Run][Settings]`. Filters: All/Text/Image/Audio/Video/Vision/Installed/Favorites. Model page tabs: Overview/Files/Requirements/Performance/Versions/Outputs/Logs. Show compat % + rec config + license + trust badge.

## 11.4 Unified Generate (capability-driven, hide runtime by default)
- Image (SD): prompt, negative prompt, steps 25, CFG 7, 512×512, [Generate].
- TTS (Kokoro): text, voice, speed 1.0, [Generate Audio] + playback.
- LLM (Qwen): prompt, temp 0.7, max-tokens 2048, [Generate] + streaming chat (v0.2).
Form schema derives from `capabilities` (`04-DATA-MODEL.md`); unknown caps -> generic JSON form, never crash.

## 11.5 Modes
Beginner (default): Choose Model -> Prompt -> Generate. Advanced: runtime/dtype/scheduler/CUDA/layers/offload/threads/quant/env. Developer: terminal, API, logs, env, python console, runtime cmd, model files.

## 11.6 Logs & error translation (differentiator)
Scopes: Runtime/Download/Model/System. Beginners see human cause + fix; `View Technical Logs` for raw. Translate e.g. `CUDA OOM` -> "Not enough GPU memory (need ~6.2GB, have 4GB). Try CPU offload / lower precision / smaller resolution / quantized model." Every ERROR maps to code + recommendation; no bare stack traces.
