//! Analyzer inputs (docs/05 §5.2): tags, `library_name`, `pipeline_tag`,
//! `model-index`, file list, `config.json` (`model_type`, `architectures`,
//! `torch_dtype`, `hidden_size`, `num_hidden_layers`) and diffusion
//! `model_index.json`. Output feeds runtime selection (docs/05 §5.5) and the
//! pre-download compat card (docs/07 §7.4).

use crate::hardware::{select_execution_mode, ExecutionMode, MemoryInfo};
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

// ---------------------------------------------------------------------------
// Registry-driven scoring (docs/05 §5.4–§5.5)
// ---------------------------------------------------------------------------
//
// The adapter registry (`runtimes/registry.json`) is the single source of
// truth for arch -> runtime mapping — never hardcoded match arms — so plugins
// can extend it without core changes. Scoring below is a pure, sync function
// of (registry, metadata, file list): unit-testable with no I/O.

/// Embedded copy of the canonical registry. Callers that need live plugin
/// entries should parse `runtimes/registry.json` at runtime and pass it to
/// [`score_candidates`]; tests and offline paths use this snapshot.
const DEFAULT_REGISTRY_JSON: &str = include_str!("../../runtimes/registry.json");

/// File-marker side of one registry mapping (docs/05 §5.2).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RegistryMarkers {
    #[serde(default)]
    pub config: Option<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub exclude_markers: Vec<String>,
    #[serde(default)]
    pub exclude_extensions: Vec<String>,
}

/// One `arch -> runtime candidates` entry (docs/05 §5.4).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RegistryMapping {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub markers: Option<RegistryMarkers>,
    #[serde(default)]
    pub runtimes: Vec<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// Unsupported-model fallback markers (docs/05 §5.6): their presence means a
/// `custom` Python env MIGHT work, offered only as opt-in Experimental Mode.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RegistryFallback {
    #[serde(default)]
    pub markers: Vec<String>,
    #[serde(default)]
    pub runtimes: Vec<String>,
}

/// Parsed `runtimes/registry.json`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AdapterRegistry {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub mappings: Vec<RegistryMapping>,
    #[serde(default)]
    pub fallback: Option<RegistryFallback>,
}

/// Parse registry JSON. Returns a descriptive string on failure — never
/// panics, so a hand-edited registry degrades to "no candidates" (which the
/// unsupported card handles) instead of crashing the analyzer.
pub fn parse_registry(json: &str) -> Result<AdapterRegistry, String> {
    serde_json::from_str(json).map_err(|e| format!("invalid adapter registry: {e}"))
}

/// The embedded registry snapshot. Parse failures yield an empty registry
/// (every model scores "unsupported with experimental offer", never a panic).
pub fn default_registry() -> AdapterRegistry {
    parse_registry(DEFAULT_REGISTRY_JSON).unwrap_or_default()
}

/// Everything the scorer needs: `config.json` architectures +
/// `model_index.json` diffusion class + file markers + `pipeline_tag`
/// (docs/05 §5.2–§5.4).
#[derive(Debug, Clone, Default)]
pub struct AnalyzerInput {
    /// `config.json` `architectures` (may be empty for diffusers-only repos).
    pub architectures: Vec<String>,
    /// `model_index.json` `_class_name` (diffusion pipelines).
    pub diffusion_class: Option<String>,
    /// Repo file list (`RepoMetadata::file_names`).
    pub files: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
}

/// Minimal `model_index.json` view: only the pipeline class matters.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DiffusionIndex {
    #[serde(default, rename = "_class_name")]
    pub class_name: Option<String>,
}

/// Extract the diffusion pipeline class from raw `model_index.json` text.
/// `None` on any parse failure or missing field (not an error — most repos
/// have no diffusion index at all).
pub fn parse_diffusion_class(json_text: &str) -> Option<String> {
    serde_json::from_str::<DiffusionIndex>(json_text)
        .ok()
        .and_then(|d| d.class_name)
        .filter(|c| !c.trim().is_empty())
}

/// One scored runtime candidate (docs/05 §5.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCandidate {
    pub runtime: String,
    /// 0–100 registry score (arch 50 + file markers 20–30 + pipeline 10 +
    /// library hint 5). Sorted desc; stable order = registry order on ties.
    pub score: u8,
    pub reason: String,
}

/// Unsupported-model card (docs/05 §5.6). This is a VALUE, not an error:
/// callers render it with an opt-in Experimental-run offer, never a bare
/// ERROR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsupportedCard {
    /// Best-effort arch label: config arch, diffusion class, pipeline tag,
    /// or `"unknown"`.
    pub detected_arch: String,
    /// Suggested escape hatch. Almost always `"custom"` (Custom Python).
    pub possible_runtime: String,
    /// Always true: the UI must offer opt-in Experimental Mode.
    pub experimental_offer: bool,
    /// Fallback setup markers actually found in the repo
    /// (`requirements.txt`, `custom_model.py`, …).
    pub setup_markers_found: Vec<String>,
    /// Human-readable fix. Mentions Advanced-mode gating + custom-code
    /// consent when `custom_model.py` is present.
    pub message: String,
}

/// Supported vs unsupported outcome of registry scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SupportStatus {
    Supported,
    Unsupported(UnsupportedCard),
}

/// Full analyzer output: classification + ranked runtime candidates +
/// pre-download compat card + hardware-aware recommendation
/// (docs/05 §5.3–§5.5, docs/07 §7.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullAnalysis {
    pub category: ModelCategory,
    pub task: String,
    pub format: ModelFormat,
    pub architectures: Vec<String>,
    pub pipeline: Option<String>,
    pub license: Option<String>,
    pub candidates: Vec<RuntimeCandidate>,
    pub compat: CompatScore,
    pub recommended: RecommendedConfig,
    pub support: SupportStatus,
}

impl FullAnalysis {
    /// Top-ranked runtime id, if any candidate scored above zero.
    pub fn recommended_runtime(&self) -> Option<&str> {
        self.candidates.first().map(|c| c.runtime.as_str())
    }

    pub fn is_supported(&self) -> bool {
        matches!(self.support, SupportStatus::Supported)
    }
}

// ---------------------------------------------------------------------------
// Hardware profile (docs/07 §7.2–§7.4). Read-only USE of the hardware
// backend's public API: profiles are built from [`MemoryInfo`] (produced by
// `CpuBackend`/`NvidiaBackend`), placement reuses [`select_execution_mode`].
// The analyzer never touches hardware itself, so scoring stays pure/testable.
// ---------------------------------------------------------------------------

/// VRAM tiers driving all quant/dtype defaults (docs/07 §7.2). The 4GB
/// GTX 1050 / 16GB RAM combo is the reference low-end target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VramTier {
    /// No discrete GPU: CPU execution path.
    CpuOnly,
    /// < 4GB.
    UltraLow,
    /// 4–6GB (reference 4GB target lives here).
    Low,
    /// 6–12GB.
    Medium,
    /// 12–24GB.
    High,
    /// 24GB+.
    Extreme,
}

impl VramTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOnly => "cpu-only",
            Self::UltraLow => "ultra-low",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Extreme => "extreme",
        }
    }
}

/// Tier by TOTAL VRAM in MB; `None` = no discrete GPU.
pub fn vram_tier(vram_total_mb: Option<u64>) -> VramTier {
    match vram_total_mb {
        None => VramTier::CpuOnly,
        Some(mb) if mb < 4 * 1024 => VramTier::UltraLow,
        Some(mb) if mb < 6 * 1024 => VramTier::Low,
        Some(mb) if mb < 12 * 1024 => VramTier::Medium,
        Some(mb) if mb < 24 * 1024 => VramTier::High,
        _ => VramTier::Extreme,
    }
}

/// VRAM/RAM snapshot the recommender needs. Build it from live hardware
/// data at the call site; tests construct it literally.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub ram_total_mb: u64,
}

impl Default for HardwareProfile {
    /// Reference low-end assumptions (docs/07 §7.2): 16GB RAM, GPU unknown.
    fn default() -> Self {
        Self {
            vram_total_mb: None,
            vram_free_mb: None,
            ram_total_mb: 16 * 1024,
        }
    }
}

impl HardwareProfile {
    /// Build from the hardware backend's memory snapshot
    /// (`CpuBackend::memory` / `NvidiaBackend::memory`).
    pub fn from_memory(mem: &MemoryInfo) -> Self {
        Self {
            vram_total_mb: mem.vram_total_mb,
            vram_free_mb: mem.vram_free_mb,
            ram_total_mb: mem.ram_total_mb,
        }
    }

    /// Reference 4GB low-end target (docs/07 §7.2): GTX 1050 4GB + 16GB RAM.
    pub fn reference_low_end() -> Self {
        Self {
            vram_total_mb: Some(4 * 1024),
            vram_free_mb: Some(4 * 1024),
            ram_total_mb: 16 * 1024,
        }
    }
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

fn norm_token(s: &str) -> String {
    s.to_lowercase().replace('_', "-")
}

/// Effective architecture list: `config.json` architectures + diffusion
/// class + pseudo-arches synthesized from weight-file markers so GGUF/ONNX
/// repos score even when `config.json` carries a generic transformers arch
/// (e.g. Qwen2-GGUF must resolve to `llama_cpp`, never `transformers`).
fn effective_arches(input: &AnalyzerInput) -> Vec<String> {
    let mut archs = input.architectures.clone();
    if let Some(d) = &input.diffusion_class {
        if !archs.iter().any(|a| a.eq_ignore_ascii_case(d)) {
            archs.push(d.clone());
        }
    }
    if input.files.iter().any(|f| f.ends_with(".gguf"))
        && !archs.iter().any(|a| a.eq_ignore_ascii_case("GGUF"))
    {
        archs.push("GGUF".to_string());
    }
    if input.files.iter().any(|f| f.ends_with(".onnx"))
        && !archs.iter().any(|a| a.eq_ignore_ascii_case("OnnxModel"))
    {
        archs.push("OnnxModel".to_string());
    }
    archs
}

fn mapping_excluded(mapping: &RegistryMapping, files: &[String]) -> bool {
    if let Some(m) = &mapping.markers {
        for x in &m.exclude_markers {
            if files
                .iter()
                .any(|f| f == x || f.ends_with(&format!("/{x}")))
            {
                return true;
            }
        }
        for x in &m.exclude_extensions {
            if files.iter().any(|f| f.ends_with(x)) {
                return true;
            }
        }
    }
    false
}

/// Score every registry mapping against the input. Weights: exact arch 50,
/// weight-file extension marker 30, distinctive config file (`config.json`
/// is ubiquitous so only 15; `model_index.json` 20) 15–20, pipeline_tag vs
/// capability 10, library_name hinting the runtime 5. A mapping counts only
/// with a STRONG signal (arch, extension, or pipeline hit) — `config.json`
/// alone must not rescue an unknown arch into `transformers`.
pub fn score_candidates(
    registry: &AdapterRegistry,
    input: &AnalyzerInput,
) -> Vec<RuntimeCandidate> {
    let archs = effective_arches(input);
    let mut out = Vec::new();
    for mapping in &registry.mappings {
        if mapping_excluded(mapping, &input.files) {
            continue;
        }
        let mut score: u32 = 0;
        let mut strong = false;
        let mut why: Vec<String> = Vec::new();
        if let Some(hit) = archs.iter().find(|a| {
            mapping
                .architectures
                .iter()
                .any(|m| m.eq_ignore_ascii_case(a))
        }) {
            score += 50;
            strong = true;
            why.push(format!("arch {hit} mapped"));
        }
        if let Some(m) = &mapping.markers {
            if m.extensions
                .iter()
                .any(|e| input.files.iter().any(|f| f.ends_with(e)))
            {
                score += 30;
                strong = true;
                why.push("weight-file marker match".to_string());
            }
            if let Some(cfg) = &m.config {
                if input
                    .files
                    .iter()
                    .any(|f| f == cfg || f.ends_with(&format!("/{cfg}")))
                {
                    // `config.json` is ubiquitous (15 pts, not qualifying);
                    // `model_index.json` exists ONLY in diffusion repos, so it
                    // qualifies on its own (20 pts, strong).
                    let w = if cfg == "model_index.json" { 20 } else { 15 };
                    if cfg == "model_index.json" {
                        strong = true;
                    }
                    score += w;
                    why.push(format!("{cfg} layout match"));
                }
            }
        }
        // Pipeline/capability overlap is a BONUS, never a qualifier on its
        // own: without an arch or weight-file hit the runtime has no
        // evidence it can read this repo (e.g. an onnx runtime must not be
        // recommended for a GGUF-only repo just because both mention text).
        if let Some(p) = &input.pipeline_tag {
            let np = norm_token(p);
            if mapping.capabilities.iter().any(|c| {
                let nc = norm_token(c);
                nc == np || np.contains(&nc) || nc.contains(&np)
            }) {
                score += 10;
                why.push(format!("pipeline {p} capability match"));
            }
        }
        if score == 0 || !strong {
            continue;
        }
        for rt in &mapping.runtimes {
            let mut s = score;
            let mut why_rt = why.clone();
            if let Some(lib) = &input.library_name {
                let nl = norm_token(lib);
                let nr = norm_token(rt);
                if !nl.is_empty() && (nl.contains(&nr) || nr.contains(&nl)) {
                    s += 5;
                    why_rt.push(format!("library {lib} hint"));
                }
            }
            out.push(RuntimeCandidate {
                runtime: rt.clone(),
                score: s.min(100) as u8,
                reason: why_rt.join("; "),
            });
        }
    }
    // Stable sort: score desc, registry order wins ties (e.g. `diffusers`
    // before `comfyui` for SD pipelines — registry order is the priority).
    out.sort_by_key(|a| std::cmp::Reverse(a.score));
    out
}

fn category_for_capabilities(caps: &[String]) -> Option<ModelCategory> {
    let joined = caps
        .iter()
        .map(|c| norm_token(c))
        .collect::<Vec<_>>()
        .join(" ");
    if joined.contains("video") {
        return Some(ModelCategory::Video);
    }
    if joined.contains("speech") || joined.contains("audio") {
        return Some(ModelCategory::Audio);
    }
    if joined.contains("image") {
        return Some(ModelCategory::Vision);
    }
    if joined.contains("text")
        || joined.contains("chat")
        || joined.contains("embedding")
        || joined.contains("classification")
    {
        return Some(ModelCategory::Text);
    }
    None
}

/// Pipeline-tag fallback when the registry yields no candidate: the same
/// keyword table the old heuristic used, minus the arch blob (unmatched
/// arch is exactly what makes the model unsupported). Hub `tags` still
/// contribute — many repos carry the task only as a tag.
fn pipeline_fallback_task(pipeline_tag: Option<&str>, tags: &[String]) -> (ModelCategory, String) {
    let blob = format!("{} {}", pipeline_tag.unwrap_or_default(), tags.join(" ")).to_lowercase();
    let has = |keys: &[&str]| keys.iter().any(|k| blob.contains(k));
    if has(&[
        "text-to-video",
        "image-to-video",
        "video-to-video",
        "video-generation",
        "video-understanding",
    ]) {
        return (
            ModelCategory::Video,
            pipeline_tag.unwrap_or("video-generation").to_string(),
        );
    }
    if has(&[
        "automatic-speech-recognition",
        "text-to-speech",
        "audio-classification",
        "audio-to-audio",
        "whisper",
    ]) {
        return (
            ModelCategory::Audio,
            pipeline_tag.unwrap_or("audio").to_string(),
        );
    }
    if has(&[
        "text-to-image",
        "image-to-image",
        "image-classification",
        "object-detection",
        "segmentation",
        "ocr",
        "stable-diffusion",
        "image-segmentation",
    ]) {
        return (
            ModelCategory::Vision,
            pipeline_tag.unwrap_or("vision").to_string(),
        );
    }
    if has(&[
        "multimodal",
        "vision-language",
        "any-to-any",
        "clip",
        "llava",
        "image-text-to-text",
    ]) {
        return (
            ModelCategory::Multimodal,
            pipeline_tag.unwrap_or("multimodal").to_string(),
        );
    }
    (
        ModelCategory::Text,
        pipeline_tag.unwrap_or("text-generation").to_string(),
    )
}

/// Registry-driven classification: score arch + diffusion class + file
/// markers + pipeline_tag, take the top mapping's capabilities for the
/// category, fall back to the pipeline keyword table when nothing scores.
/// Returns `(category, task, candidates)`.
pub fn classify_with_registry(
    registry: &AdapterRegistry,
    input: &AnalyzerInput,
) -> (ModelCategory, String, Vec<RuntimeCandidate>) {
    let candidates = score_candidates(registry, input);
    if !candidates.is_empty() {
        let top = &candidates[0];
        let top_mapping = registry.mappings.iter().find(|m| {
            m.runtimes.iter().any(|r| r == &top.runtime)
                && effective_arches(input)
                    .iter()
                    .any(|a| m.architectures.iter().any(|x| x.eq_ignore_ascii_case(a)))
        });
        if let Some(mapping) = top_mapping {
            let category =
                category_for_capabilities(&mapping.capabilities).unwrap_or(ModelCategory::Text);
            let task = input.pipeline_tag.clone().unwrap_or_else(|| {
                mapping
                    .capabilities
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "text-generation".into())
            });
            return (category, task, candidates);
        }
        // Scored on pipeline/file markers alone (no arch hit): trust the
        // pipeline tag over the winning runtime's capabilities.
        let (category, task) = pipeline_fallback_task(input.pipeline_tag.as_deref(), &input.tags);
        return (category, task, candidates);
    }
    let (category, task) = pipeline_fallback_task(input.pipeline_tag.as_deref(), &input.tags);
    (category, task, candidates)
}

/// Heuristic over pipeline tag + library + tags + architectures.
///
/// Registry-driven since the analyzer upgrade: consults the embedded
/// `runtimes/registry.json` snapshot (arch + pipeline/capability signals)
/// and falls back to the pipeline keyword table. For the full signal set
/// (file markers, diffusion class) use [`classify_with_registry`].
pub fn classify_category(
    pipeline_tag: Option<&str>,
    library_name: Option<&str>,
    tags: &[String],
    architectures: &[String],
) -> (ModelCategory, String) {
    let input = AnalyzerInput {
        architectures: architectures.to_vec(),
        diffusion_class: None,
        files: Vec::new(),
        pipeline_tag: pipeline_tag.map(str::to_string),
        library_name: library_name.map(str::to_string),
        tags: tags.to_vec(),
    };
    let registry = default_registry();
    let (category, task, _) = classify_with_registry(&registry, &input);
    (category, task)
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

/// Hardware-aware quant/dtype/offload recommendation (docs/07 §7.4,
/// docs/05 §5.5). Pure function of (profile, model size, format, runtime):
/// the caller builds `profile` from the hardware backend, so this stays
/// sync and unit-testable. Placement reuses the backend's
/// [`select_execution_mode`]; tier defaults follow docs/07 §7.2 with the 4GB
/// reference target (`Q4_K_M @4GB`, SD1.5 `FP16 + CPU offload @4GB`).
pub fn recommend_config(
    profile: &HardwareProfile,
    size_bytes: u64,
    format: ModelFormat,
    runtime: &str,
) -> RecommendedConfig {
    let need_mb = size_bytes / 1024 / 1024;
    // FP16 weights ≈ 1.2x bytes at runtime with overhead; GGUF callers pass
    // the quantized file size, which only over-estimates slightly.
    let est_vram_mb = need_mb.saturating_mul(12) / 10;
    let free = profile.vram_free_mb.or(profile.vram_total_mb);
    let offload_needed = select_execution_mode(est_vram_mb, free) != ExecutionMode::Gpu;
    let rt = runtime.to_lowercase().replace('-', "_");
    let is_llama = format == ModelFormat::Gguf || rt == "llama_cpp" || rt == "llama.cpp";
    let is_diff = format == ModelFormat::DiffusersFolder || rt == "diffusers" || rt == "comfyui";
    let is_onnx = format == ModelFormat::Onnx || rt == "onnx";

    if is_onnx {
        return RecommendedConfig {
            dtype: "fp32".to_string(),
            quant: "none".to_string(),
            offload: false,
            reason: "ONNX executes on CPU (or DirectML where available); no GPU offload or quant variant to pick"
                .to_string(),
        };
    }
    match free {
        // No discrete GPU: CPU path handles it, possibly slowly (docs/07 §7.3).
        None => {
            if is_llama {
                RecommendedConfig {
                    dtype: "q4".to_string(),
                    quant: "Q4_K_M".to_string(),
                    offload: false,
                    reason: "no discrete GPU (docs/07 CPU path): Q4_K_M GGUF via llama.cpp keeps RAM pressure down; expect slow CPU inference".to_string(),
                }
            } else if is_diff {
                RecommendedConfig {
                    dtype: "fp32".to_string(),
                    quant: "none".to_string(),
                    offload: false,
                    reason: "no discrete GPU: diffusers/comfyui on CPU in fp32 (slow — SD-class is minutes per image); prefer an NVIDIA host for image runs".to_string(),
                }
            } else {
                RecommendedConfig {
                    dtype: "fp16".to_string(),
                    quant: "fp16".to_string(),
                    offload: false,
                    reason: "fits VRAM; offload only if estimate exceeds free VRAM; no discrete GPU detected — CPU execution".to_string(),
                }
            }
        }
        // 4GB-class GPU (docs/07 Ultra-low/Low): quantize + offload.
        Some(v) if v <= 6 * 1024 => {
            if is_llama {
                RecommendedConfig {
                    dtype: "fp16".to_string(),
                    quant: "Q4_K_M".to_string(),
                    offload: true,
                    reason: "4GB-class GPU (docs/07 Low, docs/05 §5.5): pick the Q4_K_M GGUF variant; llama.cpp offloads partial layers to GPU, rest stays on CPU".to_string(),
                }
            } else if is_diff {
                RecommendedConfig {
                    dtype: "fp16".to_string(),
                    quant: "none".to_string(),
                    offload: true,
                    reason: "SD-class model on 4GB (docs/05 §5.5): diffusers/comfyui in fp16 with CPU offload + attention slicing/VAE tiling".to_string(),
                }
            } else {
                RecommendedConfig {
                    dtype: "fp16".to_string(),
                    quant: "Q4_K_M".to_string(),
                    offload: true,
                    reason: "4GB-class GPU: quantized weights + CPU offload".to_string(),
                }
            }
        }
        // Larger VRAM: quant scales down with the tier; offload only when the
        // estimate exceeds free VRAM.
        Some(_) => {
            if is_llama {
                let quant = match vram_tier(profile.vram_total_mb) {
                    VramTier::Medium => "Q5_K_M",
                    VramTier::High | VramTier::Extreme => "Q8_0",
                    _ => "Q4_K_M",
                };
                RecommendedConfig {
                    dtype: "fp16".to_string(),
                    quant: quant.to_string(),
                    offload: offload_needed,
                    reason: format!(
                        "fits VRAM ({} tier); {quant} balances quality vs headroom; offload only if estimate exceeds free VRAM",
                        vram_tier(profile.vram_total_mb).as_str()
                    ),
                }
            } else if is_diff {
                RecommendedConfig {
                    dtype: "fp16".to_string(),
                    quant: "none".to_string(),
                    offload: offload_needed,
                    reason: "fits VRAM; fp16 with offload only if estimate exceeds free VRAM"
                        .to_string(),
                }
            } else {
                RecommendedConfig {
                    dtype: "fp16".to_string(),
                    quant: "fp16".to_string(),
                    offload: offload_needed,
                    reason: "fits VRAM; offload only if estimate exceeds free VRAM".to_string(),
                }
            }
        }
    }
}

/// Pre-download compat estimate: est. RAM/VRAM vs user RAM/VRAM, verdict
/// Good/Poor — blocks wasteful downloads (docs/07 §7.4).
///
/// GPU subscore reuses the hardware backend's [`select_execution_mode`]
/// (fits → 95, offload window → 55, else 10; CPU-only machines score 60 and
/// always pass — the CPU path is slow, not an error). The recommendation
/// delegates to [`recommend_config`] with an unknown format, preserving the
/// historical `Q4_K_M @<=6GB` / `fp16` table.
pub fn compat_score(
    size_bytes: u64,
    user_ram_mb: u64,
    user_vram_mb: Option<u64>,
    runtime_ready: bool,
) -> (CompatScore, RecommendedConfig) {
    let need_mb = size_bytes / 1024 / 1024;
    // FP16 weights ≈ 2x bytes at runtime with overhead; GGUF handled by caller.
    let est_vram_mb = need_mb.saturating_mul(12) / 10;
    let gpu = match (
        user_vram_mb,
        select_execution_mode(est_vram_mb, user_vram_mb),
    ) {
        (None, _) => 60,
        (Some(_), ExecutionMode::Gpu) => 95,
        (Some(_), ExecutionMode::Offload) => 55,
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
    let profile = HardwareProfile {
        vram_total_mb: user_vram_mb,
        vram_free_mb: user_vram_mb,
        ram_total_mb: user_ram_mb,
    };
    let recommended = recommend_config(&profile, size_bytes, ModelFormat::Unknown, "");
    (
        CompatScore {
            overall,
            gpu,
            ram,
            runtime,
            verdict,
        },
        recommended,
    )
}

/// Build the unsupported-model card (docs/05 §5.6): detected arch, the
/// `custom` escape hatch, fallback setup markers found in the repo, and an
/// explicit opt-in Experimental-run offer gated behind Advanced mode +
/// custom-code consent. Never a bare error.
pub fn unsupported_card(registry: &AdapterRegistry, input: &AnalyzerInput) -> UnsupportedCard {
    let detected = input
        .architectures
        .first()
        .or(input.diffusion_class.as_ref())
        .cloned()
        .or_else(|| input.pipeline_tag.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let fallback_markers: &[String] = registry
        .fallback
        .as_ref()
        .map(|f| f.markers.as_slice())
        .unwrap_or(&[]);
    let setup_markers_found: Vec<String> = fallback_markers
        .iter()
        .filter(|m| {
            input
                .files
                .iter()
                .any(|f| f == *m || f.ends_with(&format!("/{m}")))
        })
        .cloned()
        .collect();
    let custom_code = setup_markers_found.iter().any(|m| m == "custom_model.py");
    let mut message = format!(
        "Detected architecture '{detected}' has no runtime mapping in runtimes/registry.json, \
         so there is no verified runtime for this model (not a download error — nothing failed). \
         The likely escape hatch is the Custom Python adapter."
    );
    if setup_markers_found.is_empty() {
        message.push_str(
            " No Python setup markers (requirements.txt / pyproject.toml / setup.py / \
             environment.yml) were found, so even a custom env would need manual dependency work.",
        );
    } else {
        message.push_str(&format!(
            " Found setup markers: {}.",
            setup_markers_found.join(", ")
        ));
    }
    message.push_str(
        " Offer: opt-in Experimental run behind Advanced mode — isolated Python+Torch+Transformers/Accelerate \
         install with explicit consent before anything executes.",
    );
    if custom_code {
        message.push_str(
            " custom_model.py is present: running it requires explicit custom-code trust consent (never auto-run).",
        );
    }
    UnsupportedCard {
        detected_arch: detected,
        possible_runtime: "custom".to_string(),
        experimental_offer: true,
        setup_markers_found,
        message,
    }
}

/// Full metadata-only analysis: no weight loading, no inference.
/// Registry scoring + hardware-aware recommendation + unsupported card in
/// one call. `size_bytes` is the Hub-reported total (0 = unknown, compat
/// degrades gracefully); `runtime_ready` is whether the top candidate's env
/// is already installed.
pub fn analyze_full(
    meta: &RepoMetadata,
    config: Option<&TransformersConfig>,
    diffusion: Option<&DiffusionIndex>,
    registry: &AdapterRegistry,
    hw: &HardwareProfile,
    size_bytes: u64,
    runtime_ready: bool,
) -> FullAnalysis {
    let files = meta.file_names();
    let mut architectures = config.map(|c| c.architectures.clone()).unwrap_or_default();
    let diffusion_class = diffusion.and_then(|d| d.class_name.clone());
    if architectures.is_empty() {
        if let Some(d) = &diffusion_class {
            architectures.push(d.clone());
        }
    }
    let input = AnalyzerInput {
        architectures: architectures.clone(),
        diffusion_class,
        files,
        pipeline_tag: meta.pipeline_tag.clone(),
        library_name: meta.library_name.clone(),
        tags: meta.tags.clone(),
    };
    let (category, task, candidates) = classify_with_registry(registry, &input);
    let format = detect_format(&meta.file_names());
    let support = if candidates.is_empty() {
        SupportStatus::Unsupported(unsupported_card(registry, &input))
    } else {
        SupportStatus::Supported
    };
    let free = hw.vram_free_mb.or(hw.vram_total_mb);
    let (compat, _) = compat_score(size_bytes, hw.ram_total_mb, free, runtime_ready);
    let recommended = recommend_config(
        hw,
        size_bytes,
        format,
        candidates.first().map(|c| c.runtime.as_str()).unwrap_or(""),
    );
    FullAnalysis {
        category,
        task,
        format,
        architectures,
        pipeline: meta.pipeline_tag.clone(),
        license: meta.license(),
        candidates,
        compat,
        recommended,
        support,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::HfFileEntry;

    fn files(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn repo_meta(pipeline_tag: Option<&str>, file_list: &[&str]) -> RepoMetadata {
        RepoMetadata {
            id: "test/model".to_string(),
            tags: vec![],
            pipeline_tag: pipeline_tag.map(str::to_string),
            library_name: None,
            license: None,
            siblings: file_list
                .iter()
                .map(|f| HfFileEntry {
                    rfilename: f.to_string(),
                    size: None,
                    lfs: None,
                })
                .collect(),
            card: None,
        }
    }

    fn sd15_input() -> AnalyzerInput {
        AnalyzerInput {
            architectures: vec!["StableDiffusionPipeline".to_string()],
            diffusion_class: Some("StableDiffusionPipeline".to_string()),
            files: files(&[
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "tokenizer/merges.txt",
                "tokenizer/vocab.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ]),
            pipeline_tag: Some("text-to-image".to_string()),
            library_name: Some("diffusers".to_string()),
            tags: vec![],
        }
    }

    #[test]
    fn registry_snapshot_parses() {
        let reg = default_registry();
        assert!(reg.mappings.len() >= 7, "expected all registry mappings");
        assert!(reg.fallback.is_some());
    }

    #[test]
    fn sd15_scores_diffusers_first() {
        let reg = default_registry();
        let input = sd15_input();
        let (category, task, candidates) = classify_with_registry(&reg, &input);
        assert_eq!(category, ModelCategory::Vision);
        assert_eq!(task, "text-to-image");
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].runtime, "diffusers");
        assert!(candidates.iter().any(|c| c.runtime == "comfyui"));
        assert!(candidates[0].score >= 80);
    }

    #[test]
    fn qwen_gguf_scores_llama_cpp_only() {
        let reg = default_registry();
        let input = AnalyzerInput {
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            diffusion_class: None,
            files: files(&[
                "config.json",
                "generation_config.json",
                "qwen2-7b-instruct-q4_k_m.gguf",
                "tokenizer.json",
            ]),
            pipeline_tag: Some("text-generation".to_string()),
            library_name: Some("transformers".to_string()),
            tags: vec![],
        };
        let (category, task, candidates) = classify_with_registry(&reg, &input);
        assert_eq!(category, ModelCategory::Text);
        assert_eq!(task, "text-generation");
        // `.gguf` excludes the transformers mapping: llama_cpp only.
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].runtime, "llama_cpp");
        assert!(candidates[0].score >= 80);
    }

    #[test]
    fn unknown_arch_is_unsupported_with_experimental_offer() {
        let reg = default_registry();
        let input = AnalyzerInput {
            architectures: vec!["HypotheticalFutureForCausalLM".to_string()],
            diffusion_class: None,
            // config.json alone is a weak signal and must not rescue it.
            files: files(&["config.json", "model.safetensors"]),
            pipeline_tag: None,
            library_name: None,
            tags: vec![],
        };
        let candidates = score_candidates(&reg, &input);
        assert!(candidates.is_empty());
        let card = unsupported_card(&reg, &input);
        assert_eq!(card.detected_arch, "HypotheticalFutureForCausalLM");
        assert_eq!(card.possible_runtime, "custom");
        assert!(card.experimental_offer);
        assert!(!card.message.is_empty());
        assert!(card.message.contains("Experimental"));
    }

    #[test]
    fn unsupported_card_surfaces_setup_markers() {
        let reg = default_registry();
        let input = AnalyzerInput {
            architectures: vec!["SomethingCustom".to_string()],
            diffusion_class: None,
            files: files(&["custom_model.py", "requirements.txt", "weights.bin"]),
            pipeline_tag: None,
            library_name: None,
            tags: vec![],
        };
        let card = unsupported_card(&reg, &input);
        assert!(card
            .setup_markers_found
            .contains(&"requirements.txt".to_string()));
        assert!(card
            .setup_markers_found
            .contains(&"custom_model.py".to_string()));
        assert!(card.message.contains("custom-code trust consent"));
    }

    #[test]
    fn hw_low_end_recommends_q4_and_offload() {
        let hw = HardwareProfile::reference_low_end();
        let rec = recommend_config(&hw, 4 * 1024 * 1024 * 1024, ModelFormat::Gguf, "llama_cpp");
        assert_eq!(rec.quant, "Q4_K_M");
        assert!(rec.offload);
        let rec = recommend_config(
            &hw,
            4 * 1024 * 1024 * 1024,
            ModelFormat::DiffusersFolder,
            "diffusers",
        );
        assert_eq!(rec.dtype, "fp16");
        assert!(rec.offload);
        assert!(rec.reason.contains("offload"));
    }

    #[test]
    fn hw_cpu_only_never_offloads() {
        let hw = HardwareProfile::default();
        let rec = recommend_config(&hw, 1024 * 1024 * 1024, ModelFormat::Gguf, "llama_cpp");
        assert!(!rec.offload);
        assert_eq!(rec.quant, "Q4_K_M");
    }

    #[test]
    fn vram_tier_boundaries() {
        assert_eq!(vram_tier(None), VramTier::CpuOnly);
        assert_eq!(vram_tier(Some(1024)), VramTier::UltraLow);
        assert_eq!(vram_tier(Some(4096)), VramTier::Low);
        assert_eq!(vram_tier(Some(6144)), VramTier::Medium);
        assert_eq!(vram_tier(Some(12288)), VramTier::High);
        assert_eq!(vram_tier(Some(24576)), VramTier::Extreme);
    }

    #[test]
    fn diffusion_class_parses() {
        let json = r#"{"_class_name": "StableDiffusionXLPipeline", "_diffusers_version": "0.30"}"#;
        assert_eq!(
            parse_diffusion_class(json),
            Some("StableDiffusionXLPipeline".to_string())
        );
        assert_eq!(parse_diffusion_class("{}"), None);
        assert_eq!(parse_diffusion_class("not json"), None);
    }

    #[test]
    fn compat_good_and_poor() {
        // 500MB model on 16GB RAM + 4GB VRAM: Good.
        let (score, _) = compat_score(500 * 1024 * 1024, 16 * 1024, Some(4 * 1024), false);
        assert_eq!(score.verdict, "Good");
        // 40GB model on 16GB RAM + 4GB VRAM: Poor.
        let (score, _) = compat_score(40 * 1024 * 1024 * 1024, 16 * 1024, Some(4 * 1024), false);
        assert_eq!(score.verdict, "Poor");
    }

    #[test]
    fn full_analysis_sd15_supported_with_recommendation() {
        let reg = default_registry();
        let meta = repo_meta(
            Some("text-to-image"),
            &[
                "model_index.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        );
        let hw = HardwareProfile::reference_low_end();
        let full = analyze_full(&meta, None, None, &reg, &hw, 4 * 1024 * 1024 * 1024, false);
        // No config arch and no diffusion JSON: pipeline-only hit still
        // classifies Vision and stays supported (never a bare error).
        assert_eq!(full.category, ModelCategory::Vision);
        assert!(full.is_supported());
        assert!(full.recommended.offload);
    }

    #[test]
    fn full_analysis_unknown_is_unsupported_never_bare_error() {
        let reg = default_registry();
        let meta = repo_meta(None, &["config.json", "model.safetensors"]);
        let hw = HardwareProfile::default();
        let config = TransformersConfig {
            model_type: None,
            architectures: vec!["HypotheticalFutureForCausalLM".to_string()],
            torch_dtype: None,
            hidden_size: None,
            num_hidden_layers: None,
        };
        let full = analyze_full(&meta, Some(&config), None, &reg, &hw, 0, false);
        assert!(!full.is_supported());
        match &full.support {
            SupportStatus::Unsupported(card) => {
                assert!(card.experimental_offer);
                assert!(!card.message.is_empty());
            }
            SupportStatus::Supported => panic!("unknown arch must not be Supported"),
        }
    }
}
