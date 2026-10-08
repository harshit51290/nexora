//! Direct calls into the `nexora` core (same fns as REST/CLI).
//! `analyze_url` is fully offline-capable: real URL parse, real
//! pre-classifier, REAL hardware-aware quant/dtype/offload recommendation
//! and compat bars from live `MemoryInfo`. Model size is unknown until the
//! Hub fetch, so the card labels the estimate as pre-download.

use nexora::analyze::{compat_score, recommend_config, HardwareProfile, ModelFormat};
use nexora::api::core_stub;
use nexora::hardware::{HardwareBackend, HardwareInfo, NvidiaBackend};

use crate::state::Analyzed;

/// Real machine snapshot (CPU baseline + `nvidia-smi` when present).
pub fn detect_hardware() -> HardwareInfo {
    NvidiaBackend::detect()
}

/// Conservative format guess from the repo NAME only (never presented as
/// evidence — the full file-marker scoring needs the Hub fetch).
fn guess_format(repo_id: &str) -> ModelFormat {
    let lower = repo_id.to_lowercase();
    if lower.contains("gguf") {
        ModelFormat::Gguf
    } else if lower.contains("onnx") {
        ModelFormat::Onnx
    } else if lower.contains("stable-diffusion") || lower.contains("sdxl") {
        ModelFormat::DiffusersFolder
    } else {
        ModelFormat::Unknown
    }
}

/// Real parse + real classify + real hardware-aware recommendation.
pub fn analyze_url(url: &str) -> Result<Analyzed, String> {
    let id = core_stub::parse_hf_url(url).map_err(|e| format!("[{}] {}", e.code, e.message))?;
    let rec = core_stub::classify_model(&id);
    let format = guess_format(&id);
    let mem = NvidiaBackend::memory();
    let profile = HardwareProfile::from_memory(&mem);
    // Size 0 = unknown until fetch; tier branches still yield the right
    // dtype/quant family, and `reason` says so.
    let cfg = recommend_config(&profile, 0, format, &rec.runtime);
    let (score, _) = compat_score(
        0,
        mem.ram_total_mb,
        mem.vram_total_mb,
        rec.runtime != "unsupported",
    );
    Ok(Analyzed {
        id,
        task: rec.task,
        runtime: rec.runtime,
        notes: rec.notes,
        quant: cfg.quant,
        dtype: cfg.dtype,
        offload: cfg.offload,
        reason: format!(
            "{} (pre-download: size unknown until Hub fetch)",
            cfg.reason
        ),
        verdict: score.verdict,
        overall: score.overall,
        gpu: score.gpu,
        ram: score.ram,
        runtime_score: score.runtime,
    })
}

/// Install / generate need the async `CoreHandle` (DB + scheduler +
/// runtimes). Coded stub until TODO-WIRE-UI — same contract as the API.
pub fn install_not_wired() -> String {
    "[E_CORE_NOT_WIRED] Install from this UI needs the async service handle. Use `uar install <owner/model>` for now.".into()
}

pub fn generate_not_wired() -> String {
    "[E_CORE_NOT_WIRED] Generate from this UI needs the scheduler + runtimes. Use `uar run` / the API for now.".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_url_is_a_coded_error() {
        let err = analyze_url("not a url").unwrap_err();
        assert!(err.starts_with("[E_BAD_HF_URL]"), "got: {err}");
    }

    #[test]
    fn sd_url_routes_to_diffusers_with_low_vram_recipe_shape() {
        let a = analyze_url("https://huggingface.co/runwayml/stable-diffusion-v1-5").unwrap();
        assert_eq!(a.runtime, "diffusers");
        assert_eq!(a.task, "text-to-image");
        assert!(!a.quant.is_empty() && !a.dtype.is_empty() && !a.reason.is_empty());
        assert!(a.overall <= 100);
    }

    #[test]
    fn gguf_url_routes_to_llama_cpp_only() {
        let a = analyze_url("https://huggingface.co/Qwen/Qwen2-7B-GGUF").unwrap();
        assert_eq!(a.runtime, "llama.cpp");
        assert!(!a.quant.is_empty());
    }

    #[test]
    fn unknown_arch_yields_unsupported_not_error() {
        let a = analyze_url("https://huggingface.co/someone/mystery-xyz-999").unwrap();
        assert_eq!(a.runtime, "unsupported");
    }
}
