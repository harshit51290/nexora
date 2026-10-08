//! Model analyzer: metadata-only classification (docs/05-HF-INTEGRATION.md).
//! Orchestration-only: reads `config.json` / `model_index.json` + file
//! extensions + tags. Never loads weights or runs inference.

pub mod analyzer;

pub use analyzer::{
    analyze_full, analyze_repo, classify_category, classify_with_registry, compat_score,
    default_registry, detect_format, parse_diffusion_class, parse_registry, recommend_config,
    score_candidates, unsupported_card, vram_tier, AdapterRegistry, AnalysisResult, AnalyzerInput,
    CompatScore, DiffusionIndex, FullAnalysis, HardwareProfile, ModelCategory, ModelFormat,
    RecommendedConfig, RegistryFallback, RegistryMapping, RegistryMarkers, RuntimeCandidate,
    SupportStatus, TransformersConfig, UnsupportedCard, VramTier,
};
