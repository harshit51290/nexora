# 05 — Hugging Face Integration & Model Analyzer

## 5.1 Import (central feature)
Input `https://huggingface.co/<owner>/<repo>`. Call HF Hub API -> repo info, file list, model card, tags, arch, framework, sizes, license, `library_name/pipeline_tag/model-index`. Support HF token (OS credential store, never plaintext DB) for private/gated repos. Must parse revisions (`@rev`).

## 5.2 Analyzer inputs
- Metadata: tags, library_name, pipeline_tag, model-index.
- Files: `*.safetensors,*.bin,*.pt,*.pth,*.gguf,*.onnx,*.ckpt` plus SD layouts (`vae/ text_encoder/ unet/ scheduler/`, `tokenizer.json`, `tokenizer_config.json`, `processor_config.json`, `README.md`, `requirements.txt`, `custom_model.py`).
- Configs: `config.json` (model_type, architectures, torch_dtype, hidden_size, num_hidden_layers), diffusion `model_index.json`.

## 5.3 Classification (5 categories — all required)
1. Text: LLM/gen, classification, embedding, reranking.
2. Vision: classification, detection, segmentation, OCR, vision-language, image-gen.
3. Audio: STT, TTS, classification, music/sound-gen, separation.
4. Video: T2V, I2V, V2V, video understanding.
5. Multimodal: image/audio/video + text combos.
Output: `{category, task, format, architectures, pipeline, license}` + compat input.

## 5.4 Adapter registry (arch -> runtime)
`LlamaForCausalLM -> transformers/llama.cpp; StableDiffusionPipeline -> diffusers; WhisperForConditionalGeneration -> transformers/whisper`. Registry is data file (`runtimes/registry.json`), not hardcoded match arms, so plugins can extend.

## 5.5 Runtime selection
`Model -> candidate runtimes -> HW compat -> perf estimate -> best runtime + config (dtype/quant/offload)`. Examples: SD1.5 on 4GB -> Diffusers/ComfyUI FP16+CPU offload; Qwen3-GGUF -> llama.cpp Q4_K_M. Show compat bars (overall/GPU/RAM/runtime %) + recommended config before download.

## 5.6 Unsupported + experimental
Never bare ERROR. Show detected arch, possible runtime (often Custom Python), offer opt-in Experimental Mode: detect `requirements.txt/pyproject.toml/setup.py/environment.yml`, propose Python+Torch+Transformers/Accelerate, isolated install, explicit custom-code consent. Gate behind Advanced mode + warning.

## 5.7 Search + licensing
Future in-app explorer: query + filters (task/size/downloads/license/format/VRAM/runtime). Always surface license with commercial-use interpretation (e.g. Apache-2.0 allowed vs research-only restricted); do not assume commercial-safe.

## 5.8 Pre-download memory estimation (hf-mem method)
`src/mem/` ports `alvarobartt/hf-mem` (MIT) to Rust: Safetensors headers
and GGUF metadata are read through HTTP Range requests — first 100KB
(then the remainder) for safetensors, 1MB doubling to 100MB for GGUF —
so weight bytes, KV-cache bytes (experimental: GQA-aware, hybrid sliding
window, MoE split), and totals are known WITHOUT downloading.
Feeds the estimate card (docs/07 §7.4), the VRAM gate, `uar estimate`,
and `POST /models/estimate`. Attribution lives in each file header.
