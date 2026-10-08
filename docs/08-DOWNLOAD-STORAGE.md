# 08 — Install, Download, Storage, Outputs

## 8.1 Install flow (8 steps, user sees 1 progress bar)
1 dir 2 env 3 download files 4 verify 5 deps 6 validate 7 test-load 8 add to library. UI: `Installing Qwen... [72%] Downloading model-00003-of-00004.safetensors`.

## 8.2 Download manager (M4)
Pause/resume/cancel/retry, parallel chunks, checksum, disk-space precheck, speed + ETA (`7.4/12.1GB 24MB/s ETA 3m18s`). Resumable large-model downloads are a release gate.

## 8.3 Storage manager
Show `Models 82GB / Runtimes 14GB / Caches 9GB / Downloads 4GB / Total 109GB`. Actions: delete model, clear cache, move models, change root dir. Dedup by content hash: shared tokenizer/VAE/text-encoder stored once, ref-counted (`Model A+B -> shared VAE`).

## 8.4 Outputs & history
Auto-organized `outputs/{Images,Audio,Video,Text,Other}/` + sidecar JSON (model/prompt/seed/params/timestamp/runtime). History entries (e.g. `#382 SDXL cyberpunk, 30 steps, CFG 7, seed 482913`) support Reuse/Regenerate/Edit/Open/Delete.
