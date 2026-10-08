# 01 — Nexora Vision, PRD, Constraints

## 1.1 Vision
Desktop app = universal launcher + runtime manager for Hugging Face models. User pastes HF URL; app figures out how to run it. No Python/CUDA/PyTorch/Diffusers/GGUF/ONNX/ComfyUI knowledge required.

UX metaphor: LM Studio + Ollama + ComfyUI + Pinokio + HF Hub + package/env manager + GPU resource manager.

## 1.2 Core problem: Model ≠ Runtime
HF repo file sets vary wildly (`config.json+safetensors`, `model.gguf`, `model.onnx`, `pytorch_model.bin`, `vae/text_encoder/unet/scheduler/`, `requirements.txt+custom_model.py`). Download ≠ runnable.

Separate concepts:
- **Model**: what the AI is.
- **Runtime**: software that executes it (Transformers, llama.cpp, Diffusers, ONNX, ComfyUI, Whisper, custom).

Pipeline: `HF Model -> Analyzer -> Runtime Selection -> Environment -> HW Optimization -> Inference -> Output`.

## 1.3 Design principle (non-negotiable)
> **Rust is orchestration. Existing runtimes are execution.**
Rust owns: downloads, discovery, metadata, detection, env mgmt, processes, GPU info, memory, cache, UI comms, logs, APIs, permissions, plugins. Inference delegated to existing runtimes. Do not build universal NN engine from scratch.

## 1.4 Platforms
- **Primary: Windows** (NVIDIA/AMD/iGPU gaming PCs, laptops; easiest for non-technical users).
- Later: Linux, then macOS. Tauri enables cross-platform UI; backend holds platform-specific runtime logic.

## 1.5 Product definition (use this, not the alternative)
✅ "Universal local AI runtime manager that auto-discovers the right inference backend."
❌ "An app that runs every HF model."
Required questions: WHAT IS THIS MODEL? WHAT CAN RUN IT? WHAT DOES HW SUPPORT? WHICH RUNTIME IS BEST? HOW TO INSTALL/CONFIGURE/RUN?

## 1.6 What NOT to build (v0.1–v1.0)
Own NN framework, CUDA impl, diffusion engine, tensor lib, every audio/video backend, cloud inference, mobile, multi-user server. Instead: Rust + existing runtimes + orchestration + UX.

## 1.7 Naming (decide later, constraints)
Avoid "Hugging Face" in name (future sources: GitHub, local files, CivitAI, ModelScope, private servers). Candidates: Forge, ModelForge, AI Forge, InferHub, ModelOS, RunAI, LocalForge, NeuralForge, TensorDock, ModelDock, InferDock, AIStation, NeuroBox, ModelBox.

## 1.8 Business model (local must work without subscription)
- Free: local models, basic runtimes, model mgmt.
- Pro (later): cloud execution, auto-conversion, advanced optimization, sync.
- Team (later): shared library, remote inference, user mgmt, private repos.

## 1.9 Risks + mitigations
1. Compatibility (1000s archs) -> adapter/plugin architecture + registry.
2. Dependency hell -> isolated envs per runtime + reuse when compatible.
3. CUDA pain -> diagnostics + known-good runtime packages + version pinning.
4. Huge downloads -> resume/streaming, storage mgr, quantization recs, pre-download estimates.
5. Custom code (`trust_remote_code`) -> warnings + sandbox + trust levels; never silent exec.
6. Low VRAM -> compat estimation + offload/FP16/quant before download.
7. Runtime rot -> version pinning + rollback + optional model updates (never silent).

## 1.10 Long-term vision
Local AI OS: Models(HF/GitHub/Local) × Runtimes(Transformers/Diffusers/llama.cpp/ONNX/ComfyUI) × Hardware(NVIDIA/AMD/Intel/CPU) -> Workflows(Image/Audio/Video) -> API(Python/Discord/Web).
