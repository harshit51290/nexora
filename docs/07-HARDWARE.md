# 07 — Hardware, VRAM, Performance

## 7.1 Detection (M2 dashboard)
CPU, RAM, GPU, VRAM, CUDA + driver, storage free, OS/arch. Display e.g. `i7 / 16GB / GTX 1050 4GB / CUDA avail / 180GB free` + `Status: Ready`.

## 7.2 Profiles (drive all defaults)
Ultra-low <4GB | Low 4–6GB | Medium 6–12GB | High 12–24GB | Extreme 24GB+. 4GB GTX 1050/16GB RAM is the reference low-end target.

## 7.3 VRAM manager (required before every load)
`Required -> Available -> fits? GPU : offload -> still impossible? CPU mode`. Auto-configure where supported: GPU/CPU offload, FP16/FP8/INT8/INT4, attention slicing, VAE tiling, low-mem mode, CPU threads, GPU layers. Memory-gated load with background progress (`allocating GPU / tokenizer / model / runtime init`).

## 7.4 Scores & estimates (before download, never after)
- Compat bars: overall/GPU/RAM/runtime %.
- Perf estimate card: model size, est. RAM/VRAM vs user RAM/VRAM, verdict Good/Poor (e.g. 70B/40GB on 16GB = Poor — blocks wasteful download).
- Quant rec: pick FP16/Q8/Q6/Q5/Q4/Q3 variant for user VRAM (4GB -> Q4_K_M + why).
- `OptimizationProfile`: dtype/quant/offload/batch/resolution/attention/VAE/threads/layers per runtime caps.
- Recommendations feed: small/quant LLMs, SD1.5, small TTS/vision for 4GB; never push 30B as "easy".
- Measured, not guessed: `src/mem/` (hf-mem port, docs/05 §5.8) reads weight/KV bytes pre-download; the install gate + compat use measured totals with heuristic fallback, and `plan_offload` turns per-block GGUF bytes into an exact `--gpu-layers` suggestion.

## 7.5 Resource management
Optional auto-unload: `Model A loaded + B requested + VRAM short -> unload A -> load B`. Concurrent models (LLM+TTS+Image) allowed only if scheduler VRAM/RAM/CPU estimate passes. See `10-API-CLI.md` for queue.
