# 10 — API, CLI, Jobs, Workflows

## 10.1 Local REST (Axum on localhost)
`GET /models /hardware /runtimes | POST /models/install /models/load /models/unload /generate`. Example: `POST /v1/generate {model,prompt,width,height}` — runner picks backend. Powers Python/JS/Discord/automation integrations.

## 10.2 OpenAI-compat (major usability)
`POST /v1/chat/completions` on `http://localhost:8000` for text models so existing apps point at local runner unchanged. Add WS streaming (M14).

## 10.3 CLI (ships with desktop app)
`uar install <owner/model> | uar run <...> | uar models | uar hardware | uar generate --model ... --prompt "..." | uar serve`. CLI calls same Rust core as Tauri. `uar estimate <owner/model> [--experimental --max-model-len --batch-size --kv-cache-dtype --gguf-file --json]` prints weight/KV bytes, dtype table, MoE + offload lines without downloading (hf-mem method). API: `POST /models/estimate`.

## 10.4 Jobs & batch
Queue: `Job 001 Image / 002 TTS / 003 LLM` with Waiting/Running/Completed/Failed/Cancelled. Batch: `100 prompts -> runner -> model -> 100 images` (datasets/thumbnails/assets). Scheduler gates concurrency on VRAM/RAM/CPU.

## 10.5 Workflows (v0.4+)
Node graphs e.g. `Prompt->LLM->prompt-enhance->Image->Upscaler->TTS->Video`. Ship templates: Text->Image, Image->Image, Text->Speech, Speech->Text, Text->Video, Image->Video, LLM->Image, LLM->TTS. Editor later; ComfyUI backs advanced paths.

## 10.6 Web UI + Discord (later)
`http://localhost:8000` serves generation UI (browser/desktop/API/CLI share backend). Discord: bot calls runner API (`/imagine cyberpunk warrior` -> image) — after Auth + M14 API stable.
