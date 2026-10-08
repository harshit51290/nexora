//! Runtime adapters — one module per backend, all implementing
//! [`RuntimeAdapter`](adapter::RuntimeAdapter).
//!
//! * UI contract: callers use **only** `prepare` → `run` (+ `stop` /
//!   `health_check`). Never branch on backend kind outside Advanced mode.
//! * Rust orchestrates; backends execute as supervised child processes.
//! * `detect()` is pure file/metadata inspection (config arch,
//!   `model_index.json`, `.gguf` / `.onnx` markers, Whisper/Kokoro markers).

pub mod adapter;
pub mod audio;
pub mod comfyui;
pub mod custom;
pub mod diffusers;
pub mod llama_cpp;
pub mod onnx;
pub mod transformers;
pub mod video;

pub use adapter::{
    codes, config_architectures, default_env_ref, has_extension, has_file, job_json,
    looks_like_cuda_oom, parse_job, pipeline_class_name, prepare_common,
    require_serve_entrypoint, supervised_command, venv_python, AdapterBase, EnvRef,
    InferenceRequest, InferenceResult, JobPayload, RuntimeAdapter, RuntimeError, RuntimeResult,
};

/// Build every built-in adapter against the same relocatable data root
/// (`UniversalRunner/{models,runtimes,environments,…}`, docs/04 §4.5).
pub fn all_adapters(data_dir: std::path::PathBuf) -> Vec<Box<dyn RuntimeAdapter>> {
    vec![
        Box::new(transformers::TransformersAdapter::new(data_dir.clone())),
        Box::new(diffusers::DiffusersAdapter::new(data_dir.clone())),
        Box::new(llama_cpp::LlamaCppAdapter::new(data_dir.clone())),
        Box::new(onnx::OnnxAdapter::new(data_dir.clone())),
        Box::new(audio::AudioAdapter::new(data_dir.clone())),
        Box::new(video::VideoAdapter::new(data_dir.clone())),
        Box::new(comfyui::ComfyUIAdapter::new(data_dir.clone())),
        Box::new(custom::CustomAdapter::new(data_dir)),
    ]
}

/// Look up one adapter by registry id (`runtimes/registry.json` keys).
/// Returns `None` for unknown ids — including future plugin ids, which are
/// resolved through the plugin system (`docs/06` §6.9), not here.
/// Callers: on `None`, fall back to
/// `crate::plugins::PluginRegistry::find_for_model` + `spawn_plugin`
/// (plugin adapters execute out-of-process, never as `RuntimeAdapter`).
pub fn adapter_for(id: &str, data_dir: std::path::PathBuf) -> Option<Box<dyn RuntimeAdapter>> {
    all_adapters(data_dir)
        .into_iter()
        .find(|adapter| adapter.id() == id)
}
