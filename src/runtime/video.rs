//! Video adapter — delegate first (`docs/06` §6.7).
//!
//! v0.1–v0.3 builds **no** pipeline engine here. Text → encoder → diffusion →
//! VAE → frames → MP4 is treated as an advanced workflow and delegated to the
//! ComfyUI adapter; this adapter detects video-capable models and routes with
//! a proper [`codes::USE_COMFYUI`] instruction instead of failing bare.

use super::adapter::*;
use crate::model::manager::ModelRecord;
use std::path::Path;

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "video";

/// Video pipeline classes (diffusers-style `_class_name` or arch names).
const VIDEO_PIPELINES: &[&str] = &[
    "TextToVideoSDPipeline",
    "AnimateDiffPipeline",
    "CogVideoXPipeline",
    "ModelScopeTextToVideoPipeline",
    "VideoDiffusionPipeline",
];

pub struct VideoAdapter {
    base: AdapterBase,
}

impl VideoAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
        }
    }
}

impl RuntimeAdapter for VideoAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "Video (delegates to ComfyUI)"
    }

    fn detect(&self, model: &ModelRecord, model_dir: &Path) -> bool {
        if config_architectures(model_dir)
            .iter()
            .any(|a| VIDEO_PIPELINES.contains(&a.as_str()))
        {
            return true;
        }
        if pipeline_class_name(model_dir)
            .map(|class| VIDEO_PIPELINES.contains(&class.as_str()))
            .unwrap_or(false)
        {
            return true;
        }
        if model
            .capabilities
            .iter()
            .any(|c| c == "video-generation")
        {
            return true;
        }
        model
            .task
            .as_deref()
            .map(|t| matches!(t, "video-generation" | "text-to-video" | "image-to-video"))
            .unwrap_or(false)
    }

    fn install(&self) -> RuntimeResult<()> {
        // Nothing to install: execution lives in the ComfyUI adapter.
        Ok(())
    }

    fn prepare(&self, model: &ModelRecord, model_dir: &Path) -> RuntimeResult<()> {
        if !model_dir.is_dir() {
            return Err(RuntimeError::model_not_found(model_dir).with_runtime(ID));
        }
        let task_video = model
            .task
            .as_deref()
            .map(|t| matches!(t, "video-generation" | "text-to-video" | "image-to-video"))
            .unwrap_or(false);
        if has_file(model_dir, "model_index.json")
            || model
                .capabilities
                .iter()
                .any(|c| c == "video-generation")
            || task_video
        {
            return Ok(());
        }
        Err(RuntimeError::unsupported(
            "video adapter needs a text-to-video pipeline or a video-generation capability",
        ))
    }

    fn run(&self, _req: InferenceRequest) -> RuntimeResult<InferenceResult> {
        // Deliberate: never execute here. The caller (scheduler / Generate
        // page) re-dispatches to the ComfyUI adapter, which owns workflow
        // generation, server supervision, and output retrieval.
        Err(RuntimeError::new(
            codes::USE_COMFYUI,
            "video generation is delegated to the ComfyUI adapter in this build",
            "Install the ComfyUI runtime from Environments, then generate again — \
             the app builds the workflow JSON and runs it for you.",
        )
        .with_runtime(ID))
    }

    fn stop(&self) -> RuntimeResult<()> {
        self.base.stop_child()
    }

    fn health_check(&self) -> RuntimeResult<()> {
        // Healthy when delegation target resolution works (= ComfyUI bundle
        // present or installable). Never spawns anything.
        Ok(())
    }
}
