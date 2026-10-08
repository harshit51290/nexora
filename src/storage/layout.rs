//! `Nexora/{models,runtimes,environments,cache,outputs/{Images,Audio,
//! Video,Text,Other},logs,plugins}` with a user-relocatable root
//! (e.g. `D:\AI\Models`). Shared blobs (tokenizer/VAE/text-encoder) are
//! stored once under `cache/blobs/<sha256>` and ref-counted.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Relocatable data root + well-known subdirectories.
#[derive(Debug, Clone)]
pub struct StorageLayout {
    pub root: PathBuf,
}

impl StorageLayout {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// OS data dir + `/Nexora`, falling back to `./Nexora`.
    pub fn default_root() -> Self {
        let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::new(base.join("Nexora"))
    }

    pub fn models(&self) -> PathBuf {
        self.root.join("models")
    }
    pub fn runtimes(&self) -> PathBuf {
        self.root.join("runtimes")
    }
    pub fn environments(&self) -> PathBuf {
        self.root.join("environments")
    }
    pub fn cache(&self) -> PathBuf {
        self.root.join("cache")
    }
    pub fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }
    pub fn plugins(&self) -> PathBuf {
        self.root.join("plugins")
    }
    pub fn outputs(&self) -> PathBuf {
        self.root.join("outputs")
    }
    pub fn output_dir(&self, kind: OutputKind) -> PathBuf {
        self.outputs().join(kind.as_str())
    }

    /// Create the full tree (idempotent).
    pub fn ensure(&self) -> std::io::Result<()> {
        for dir in [
            self.models(),
            self.runtimes(),
            self.environments(),
            self.cache(),
            self.cache().join("blobs"),
            self.logs(),
            self.plugins(),
            self.output_dir(OutputKind::Images),
            self.output_dir(OutputKind::Audio),
            self.output_dir(OutputKind::Video),
            self.output_dir(OutputKind::Text),
            self.output_dir(OutputKind::Other),
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }
}

/// Auto-organized output buckets (docs/08 §8.4).
#[derive(Debug, Clone, Copy)]
pub enum OutputKind {
    Images,
    Audio,
    Video,
    Text,
    Other,
}

impl OutputKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Images => "Images",
            Self::Audio => "Audio",
            Self::Video => "Video",
            Self::Text => "Text",
            Self::Other => "Other",
        }
    }
}

/// SHA-256 hex of in-memory bytes (dedup key for shared blobs).
pub fn content_hash_hex(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes))
}

/// Canonical cache location for a content hash: stored once, ref-counted
/// by `model_files.dedup_hash` (docs/08 §8.3: `Model A+B -> shared VAE`).
pub fn dedup_path(cache_dir: &Path, hash_hex: &str) -> PathBuf {
    cache_dir.join("blobs").join(hash_hex)
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
