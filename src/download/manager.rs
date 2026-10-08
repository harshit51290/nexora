//! Resumable large-model downloads — a release gate (docs/08 §8.2).
//! UI shows one line per active file:
//! `7.4/12.1GB 24MB/s ETA 3m18s`.

use crate::core::{NexoraError, Result};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::io::AsyncWriteExt;

/// Live counters for one file download (mirrored to `downloads` table).
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub model_id: String,
    pub file: String,
    pub bytes_done: u64,
    pub bytes_total: Option<u64>,
    pub bytes_per_sec: f64,
}

impl DownloadProgress {
    pub fn fraction(&self) -> Option<f64> {
        self.bytes_total
            .filter(|t| *t > 0)
            .map(|t| (self.bytes_done as f64 / t as f64).clamp(0.0, 1.0))
    }
    pub fn eta_secs(&self) -> Option<u64> {
        let total = self.bytes_total?;
        if self.bytes_per_sec <= 0.0 || self.bytes_done >= total {
            return None;
        }
        Some(((total - self.bytes_done) as f64 / self.bytes_per_sec) as u64)
    }
}

/// `7.4/12.1GB 24MB/s ETA 3m18s` (docs/08 §8.2).
pub fn format_progress_line(p: &DownloadProgress) -> String {
    let eta = p.eta_secs().map_or("--".into(), format_eta);
    format!(
        "{}/{} {} ETA {}",
        fmt_bytes(p.bytes_done),
        p.bytes_total.map_or("?".into(), fmt_bytes),
        fmt_speed(p.bytes_per_sec),
        eta
    )
}

pub fn fmt_bytes(b: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let f = b as f64;
    if f >= GB {
        format!("{:.1}GB", f / GB)
    } else if f >= MB {
        format!("{:.0}MB", f / MB)
    } else {
        format!("{:.0}KB", f / 1024.0)
    }
}

fn fmt_speed(bps: f64) -> String {
    if bps >= 1024.0 * 1024.0 {
        format!("{:.0}MB/s", bps / 1024.0 / 1024.0)
    } else {
        format!("{:.0}KB/s", bps / 1024.0)
    }
}

fn format_eta(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of in-memory bytes (files use [`DownloadManager::verify_file`]).
pub fn verify_sha256_bytes(bytes: &[u8], expected_hex: &str) -> Result<()> {
    let actual = hex(Sha256::digest(bytes));
    if actual.eq_ignore_ascii_case(expected_hex.trim()) {
        Ok(())
    } else {
        Err(NexoraError::HashMismatch {
            file: "<bytes>".into(),
            expected: expected_hex.into(),
            actual,
        })
    }
}

/// Orchestrates file downloads with resume + pause/cancel + checksum.
///
/// MVP downloads sequentially with `Range` resume. N-way parallel ranged
/// chunks reuse the same offset logic once a host proves range support;
/// `parallel_chunks` records the configured width (docs/08 §8.2).
#[derive(Debug, Clone)]
pub struct DownloadManager {
    pub parallel_chunks: usize,
    paused: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
}

impl Default for DownloadManager {
    fn default() -> Self {
        Self::new(4)
    }
}

impl DownloadManager {
    pub fn new(parallel_chunks: usize) -> Self {
        Self {
            parallel_chunks: parallel_chunks.max(1),
            paused: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
    }
    pub fn resume(&self) {
        self.paused.store(false, Ordering::SeqCst);
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }
    pub fn reset(&self) {
        self.paused.store(false, Ordering::SeqCst);
        self.cancelled.store(false, Ordering::SeqCst);
    }

    /// Disk-space precheck — runs before the first byte (E_SPACE_LOW).
    pub async fn precheck_space(&self, dest_dir: &Path, required_bytes: u64) -> Result<()> {
        let free = free_disk_bytes();
        if free < required_bytes {
            return Err(NexoraError::SpaceLow {
                path: dest_dir.display().to_string(),
                required_bytes,
                free_bytes: free,
            });
        }
        Ok(())
    }

    /// Download with `Range` resume: existing `<dest>.part` size becomes the
    /// resume offset; the completed file is atomically renamed to `dest`.
    pub async fn download_url(
        &self,
        url: &str,
        dest: &Path,
        expected_sha256: Option<&str>,
    ) -> Result<PathBuf> {
        let part = dest.with_extension("part");
        let done = tokio::fs::metadata(&part)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        let client = reqwest::Client::builder()
            .user_agent("nexora/0.1")
            .build()
            .unwrap_or_default();
        let mut req = client.get(url);
        if done > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={done}-"));
        }
        let mut res = req.send().await.map_err(NexoraError::Http)?;
        if res.status() == reqwest::StatusCode::REQUESTED_RANGE_NOT_SATISFIABLE {
            // Server says the offset is past EOF — restart from scratch.
            tokio::fs::remove_file(&part).await.ok();
            res = client.get(url).send().await.map_err(NexoraError::Http)?;
        }
        let mut res = res.error_for_status().map_err(NexoraError::Http)?;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .truncate(done == 0)
            .append(done > 0)
            .write(true)
            .open(&part)
            .await?;
        let mut hasher = expected_sha256.is_some().then(Sha256::new);
        // `Response::chunk()` needs no extra stream-combinator deps.
        while let Some(chunk) = res.chunk().await.map_err(NexoraError::Http)? {
            if self.cancelled.load(Ordering::SeqCst) {
                return Err(NexoraError::Other(anyhow::anyhow!("download cancelled")));
            }
            while self.paused.load(Ordering::SeqCst) {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            file.write_all(&chunk).await?;
            if let Some(h) = hasher.as_mut() {
                h.update(&chunk);
            }
        }
        file.flush().await?;
        drop(file);
        if let (Some(h), Some(exp)) = (hasher, expected_sha256) {
            let actual = hex(h.finalize());
            if !actual.eq_ignore_ascii_case(exp.trim()) {
                return Err(NexoraError::HashMismatch {
                    file: dest.display().to_string(),
                    expected: exp.to_string(),
                    actual,
                });
            }
        }
        tokio::fs::rename(&part, dest).await?;
        Ok(dest.to_path_buf())
    }

    /// Verify an on-disk file against an expected hex digest.
    pub async fn verify_file(&self, path: &Path, expected_hex: &str) -> Result<()> {
        let bytes = tokio::fs::read(path).await?;
        let actual = hex(Sha256::digest(&bytes));
        if actual.eq_ignore_ascii_case(expected_hex.trim()) {
            Ok(())
        } else {
            Err(NexoraError::HashMismatch {
                file: path.display().to_string(),
                expected: expected_hex.to_string(),
                actual,
            })
        }
    }
}

fn free_disk_bytes() -> u64 {
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();
    // Best-effort max across drives; per-path refinement lives with the
    // storage manager (relocatable roots may sit on any drive).
    disks
        .list()
        .iter()
        .map(|d| d.available_space())
        .max()
        .unwrap_or(u64::MAX)
}
