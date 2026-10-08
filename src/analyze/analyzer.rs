//! Analyzer inputs (docs/05 §5.2): tags, `library_name`, `pipeline_tag`,
//! `model-index`, file list, `config.json` (`model_type`, `architectures`,
//! `torch_dtype`, `hidden_size`, `num_hidden_layers`) and diffusion
//! `model_index.json`. Output feeds runtime selection (docs/05 §5.5) and the
//! pre-download compat card (docs/07 §7.4).

use crate::hf::RepoMetadata;
use serde::{Deserialize, Serialize};

/// The 5 required classification categories (docs/05 §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelCategory {
    Text,
    Vision,
    Audio,
    Video,
    Multimodal,
}

impl ModelCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Multimodal => "multimodal",
        }
    }
}

/// Weight/layout format detected from file extensions + SD folder layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Safetensors,
    PytorchBin,
    Gguf,
    Onnx,
    Ckpt,
    DiffusersFolder,
    Unknown,
}

impl ModelFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::PytorchBin => "pytorch-bin",
            Self::Gguf => "gguf",
            Self::Onnx => "onnx",
            Self::Ckpt => "ckpt",
            Self::DiffusersFolder => "diffusers-folder",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TransformersConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub hidden_size: Option<u64>,
    #[serde(default)]
    pub num_hidden_layers: Option<u64>,
}

/// `{category, task, format, architectures, pipeline, license}` + compat input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub category: ModelCategory,
    pub task: String,
    pub format: ModelFormat,
    pub architectures: Vec<String>,
    pub pipeline: Option<String>,
    pub license: Option<String>,
}

/// Compat bars shown BEFORE download (docs/07 §7.4): overall/GPU/RAM/runtime %.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatScore {
    pub overall: u8,
    pub gpu: u8,
    pub ram: u8,
    pub runtime: u8,
    pub verdict: String,
}

/// `OptimizationProfile` subset per runtime caps (docs/07 §7.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedConfig {
    pub dtype: String,
    pub quant: String,
    pub offload: bool,
    pub reason: String,
}

/// Extension + SD-layout detection (docs/05 §5.2).
pub fn detect_format(files: &[String]) -> ModelFormat {
    let has = |suffix: &str| files.iter().any(|f| f.ends_with(suffix));
    let has_dir = |dir: &str| files.iter().any(|f| f.starts_with(dir));
    if has("model_index.json") && (has_dir("unet/") || has_dir("transformer/")) {
        return ModelFormat::DiffusersFolder;
    }
    if has(".gguf") {
        return ModelFormat::Gguf;
    }
    if has(".onnx") {
        return ModelFormat::Onnx;
    }
    if has(".safetensors") {
        return ModelFormat::Safetensors;
    }
    if has(".bin") || has(".pt") || has(".pth") {
        return ModelFormat::PytorchBin;
    }
    if has(".ckpt") {
        return ModelFormat::Ckpt;
    }
    ModelFormat::Unknown
}

/// Heuristic over pipeline tag + library + tags + architectures.
pub fn classify_category(
    pipeline_tag: Option<&str>,
    library_name: Option<&str>,
    tags: &[String],
    architectures: &[String],
) -> (ModelCategory, String) {
    let blob = format!(
        "{} {} {} {}",
        pipeline_tag.unwrap_or_default(),
        library_name.unwrap_or_default(),
        tags.join(" "),
        architectures.join(" ")
    )
    .to_lowercase();

    let multi = [
        "multimodal",
        "vision-language",
        "any-to-any",
        "clip",
        "llava",
    ]
    .iter()
    .any(|k| blob.contains(k));
    if multi {
        return (
            ModelCategory::Multimodal,
            pipeline_tag.unwrap_or("multimodal").into(),
        );
    }
    for key in [
        "text-to-video",
        "image-to-video",
        "video-to-video",
        "video-generation",
        "video-understanding",
    ] {
        if blob.contains(key) {
            return (
                ModelCategory::Video,
                pipeline_tag.unwrap_or("video-generation").into(),
            );
        }
    }
    for key in [
        "automatic-speech-recognition",
        "text-to-speech",
        "audio-classification",
        "audio-to-audio",
        "whisper",
    ] {
        if blob.contains(key) {
            return (ModelCategory::Audio, pipeline_tag.unwrap_or("audio").into());
        }
    }
    for key in [
        "text-to-image",
        "image-to-image",
        "image-classification",
        "object-detection",
        "segmentation",
        "ocr",
        "stable-diffusion",
        "image-segmentation",
    ] {
        if blob.contains(key) {
            return (
                ModelCategory::Vision,
                pipeline_tag.unwrap_or("vision").into(),
            );
        }
    }
    (
        ModelCategory::Text,
        pipeline_tag.unwrap_or("text-generation").into(),
    )
}

/// Full metadata-only analysis: no weight loading, no inference.
pub fn analyze_repo(meta: &RepoMetadata, config: Option<&TransformersConfig>) -> AnalysisResult {
    let files = meta.file_names();
    let format = detect_format(&files);
    let architectures = config.map(|c| c.architectures.clone()).unwrap_or_default();
    let (category, task) = classify_category(
        meta.pipeline_tag.as_deref(),
        meta.library_name.as_deref(),
        &meta.tags,
        &architectures,
    );
    AnalysisResult {
        category,
        task,
        format,
        architectures,
        pipeline: meta.pipeline_tag.clone(),
        license: meta.license(),
    }
}

/// Pre-download compat estimate: est. RAM/VRAM vs user RAM/VRAM, verdict
/// Good/Poor — blocks wasteful downloads (docs/07 §7.4).
pub fn compat_score(
    size_bytes: u64,
    user_ram_mb: u64,
    user_vram_mb: Option<u64>,
    runtime_ready: bool,
) -> (CompatScore, RecommendedConfig) {
    let need_mb = size_bytes / 1024 / 1024;
    // FP16 weights ≈ 2x bytes at runtime with overhead; GGUF handled by caller.
    let est_vram_mb = need_mb.saturating_mul(12) / 10;
    let gpu = match user_vram_mb {
        None => 60,
        Some(free) if est_vram_mb <= free => 95,
        Some(free) if est_vram_mb <= free + 8 * 1024 => 55,
        _ => 10,
    };
    let ram = if need_mb * 2 <= user_ram_mb { 90 } else { 20 };
    let runtime = if runtime_ready { 90 } else { 40 };
    let overall = ((gpu as u32 + ram as u32 + runtime as u32) / 3).min(100) as u8;
    let verdict = if overall >= 70 {
        "Good".into()
    } else {
        "Poor".into()
    };
    let (quant, dtype, offload, reason) = match user_vram_mb {
        Some(v) if v <= 6 * 1024 => (
            "Q4_K_M".to_string(),
            "fp16".to_string(),
            true,
            "4GB-class GPU: quantized weights + CPU offload".to_string(),
        ),
        _ => (
            "fp16".to_string(),
            "fp16".to_string(),
            est_vram_mb > user_vram_mb.unwrap_or(u64::MAX),
            "fits VRAM; offload only if estimate exceeds free VRAM".to_string(),
        ),
    };
    (
        CompatScore {
            overall,
            gpu,
            ram,
            runtime,
            verdict,
        },
        RecommendedConfig {
            dtype,
            quant,
            offload,
            reason,
        },
    )
}
