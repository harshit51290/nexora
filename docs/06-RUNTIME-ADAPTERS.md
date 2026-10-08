# 06 — Runtime Adapters & Plugins

## 6.1 Common trait (all runtimes implement; app calls only this)
```rust
trait RuntimeAdapter {
  fn detect(&self, model: &Model) -> bool;
  fn install(&self) -> anyhow::Result<()>;
  fn prepare(&self, model: &Model) -> anyhow::Result<()>;
  fn run(&self, req: InferenceRequest) -> anyhow::Result<InferenceResult>;
  fn stop(&self) -> anyhow::Result<()>;
  fn health_check(&self) -> anyhow::Result<()>;
}
```
Layout: `src/runtime/{transformers,diffusers,llama_cpp,onnx,comfyui,whisper,custom}/`. Flow is always `Runtime -> prepare -> run`; UI never branches on backend except Advanced mode.

## 6.2 Transformers (first adapter)
Isolated envs under `runtimes/transformers/{python,packages,environment.json}`. Chain `Python -> Transformers -> PyTorch -> CUDA`. Never touch user global Python. MVP tasks: text-gen (+classification/embedding/vision passthrough).

## 6.3 Diffusers
Handles `model_index.json` pipelines: SD/SDXL/Flux + image editing + (later) video. Detect pipeline availability from index; low-VRAM flags (FP16, attention slicing, VAE tiling) auto-applied per HW profile.

## 6.4 llama.cpp (GGUF)
Hide GGUF jargon: show Quantization (Q4_K_M default on 4GB), est. RAM, GPU offload layers, expected perf. Recommendation engine picks Q8/Q6/Q5/Q4/Q3 variant balancing VRAM vs quality.

## 6.5 ONNX (v0.5)
`.onnx` without full PyTorch. Providers: CPU, CUDA, DirectML (Windows priority). Key for AMD/Intel/iGPU later.

## 6.6 Audio (adapter layer, no single engine assumed)
`audio/{speech_to_text,text_to_speech,music,sound_generation,audio_processing}/`. Map: Whisper->STT, Kokoro->TTS, etc. v0.3: Whisper + TTS + playback.

## 6.7 Video (delegate first)
Treat as advanced workflow: Text->encoder->diffusion->VAE->frames->MP4. v0.1–v0.3: delegate complex video to ComfyUI only; do not build pipeline engine yet.

## 6.8 ComfyUI integration (killer feature)
`Universal Runner -> ComfyUI Manager -> ComfyUI Server -> Workflow -> GPU`. App auto: installs ComfyUI + required nodes, downloads models, generates workflow JSON, launches server (child proc), executes, retrieves output. Needed for M13 advanced image/video.

## 6.9 Plugin system (M15)
`plugins/<name>/{manifest,entrypoint}`. Manifest: `{name,version,supported_models[],entrypoint}`. Third-party adapters (whisper/kokoro/ollama/custom) extend registry without core changes. Sandboxed child-process exec; versioned API.
