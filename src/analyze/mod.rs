//! Model analyzer: metadata-only classification (docs/05-HF-INTEGRATION.md).
//! Orchestration-only: reads `config.json` / `model_index.json` + file
//! extensions + tags. Never loads weights or runs inference.

pub mod analyzer;

pub use analyzer::{
    analyze_repo, classify_category, compat_score, detect_format, AnalysisResult, CompatScore,
    ModelCategory, ModelFormat, RecommendedConfig,
};
