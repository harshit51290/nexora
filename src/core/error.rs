//! NEXORA_ERROR catalog: every ERROR carries a machine-readable code +
//! a human recommendation (docs/04-DATA-MODEL.md §4.2, AGENTS.md rule 7).

use thiserror::Error;

/// Convenience alias used across the orchestrator.
pub type Result<T, E = NexoraError> = std::result::Result<T, E>;

/// Machine-readable error catalog. `code()` is persisted to `logs`/state
/// rows; `human_fix()` is shown in the UI next to it.
#[derive(Debug, Error)]
pub enum NexoraError {
    /// Model needs more VRAM than currently available.
    #[error("E_VRAM_SHORT: need {required_mb}MB VRAM, only {available_mb}MB free")]
    VramShort { required_mb: u64, available_mb: u64 },

    /// CUDA allocation failed at load/run time.
    #[error("E_CUDA_OOM: CUDA out of memory while {op}")]
    CudaOom { op: String },

    /// Disk-space precheck failed before download/install.
    #[error("E_SPACE_LOW: need {required_bytes} bytes, only {free_bytes} free at {path}")]
    SpaceLow {
        path: String,
        required_bytes: u64,
        free_bytes: u64,
    },

    /// Checksum mismatch after download.
    #[error("E_HASH_MISMATCH: {file}: expected {expected}, got {actual}")]
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    /// Repo ships custom executable code; explicit consent required.
    #[error("E_CUSTOM_CODE: {repo} ships custom code ({detail}); consent required")]
    CustomCode { repo: String, detail: String },

    /// Supervised runtime child process crashed; app stays alive.
    #[error("E_RUNTIME_CRASH: runtime {runtime} (pid {pid:?}) exited: {detail}")]
    RuntimeCrash {
        runtime: String,
        pid: Option<u32>,
        detail: String,
    },

    /// Hub has no such model (or it is private/gated and no token was sent).
    #[error("E_MODEL_NOT_FOUND: unknown model {name}")]
    UnknownModel { name: String },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    #[error("db: {0}")]
    Db(#[from] sqlx::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("other: {0}")]
    Other(#[from] anyhow::Error),
}

impl NexoraError {
    /// Stable machine-readable code persisted with every ERROR state.
    pub fn code(&self) -> &'static str {
        match self {
            Self::VramShort { .. } => "E_VRAM_SHORT",
            Self::CudaOom { .. } => "E_CUDA_OOM",
            Self::SpaceLow { .. } => "E_SPACE_LOW",
            Self::HashMismatch { .. } => "E_HASH_MISMATCH",
            Self::CustomCode { .. } => "E_CUSTOM_CODE",
            Self::RuntimeCrash { .. } => "E_RUNTIME_CRASH",
            Self::UnknownModel { .. } => "E_MODEL_NOT_FOUND",
            Self::Io(_) => "E_IO",
            Self::Http(_) => "E_HTTP",
            Self::Db(_) => "E_DB",
            Self::Json(_) => "E_JSON",
            Self::Other(_) => "E_OTHER",
        }
    }

    /// Human recommendation shown in the UI next to the code.
    pub fn human_fix(&self) -> String {
        match self {
            Self::VramShort { .. } => "Pick the smaller / quantized (Q4) variant, enable CPU offload, or unload the other loaded model first.".into(),
            Self::CudaOom { .. } => "Lower batch size / resolution, switch to FP16 or INT4 quant, enable VAE tiling + attention slicing, then retry.".into(),
            Self::SpaceLow { .. } => "Free disk space, clear cache (Settings → Storage), or relocate the data root to a larger drive.".into(),
            Self::HashMismatch { .. } => "The download is corrupt or was updated upstream. Delete the partial file and retry resume; if it repeats, re-install the model.".into(),
            Self::CustomCode { .. } => "Review the files via [View Files]. Only continue with [Run in Sandbox] if you trust the publisher; otherwise Cancel.".into(),
            Self::RuntimeCrash { .. } => "The runtime child process crashed but Nexora is still running. Check Logs, update/reinstall the runtime env, then retry.".into(),
            Self::UnknownModel { .. } => "Check the owner/model spelling (or pass a full huggingface.co URL). Private or gated repos need HF_TOKEN set in the environment, then retry.".into(),
            Self::Io(e) => format!("Filesystem error ({e}). Check permissions and that the data drive is still mounted."),
            Self::Http(e) => format!("Network error ({e}). Check connection/proxy and retry resume."),
            Self::Db(e) => format!("Database error ({e}). Restart Nexora; if it persists, restore settings.db from backup."),
            Self::Json(e) => format!("Metadata parse error ({e}). The repo config is unexpected — try Experimental Mode or report the repo URL."),
            Self::Other(e) => format!("{e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_model_carries_not_found_code_and_actionable_fix() {
        let e = NexoraError::UnknownModel {
            name: "nope/missing".into(),
        };
        assert_eq!(e.code(), "E_MODEL_NOT_FOUND");
        assert!(e.to_string().contains("nope/missing"));
        assert!(e.human_fix().contains("HF_TOKEN"));
    }
}
