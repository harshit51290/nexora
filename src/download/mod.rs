//! Download manager: range-resume, parallel chunks, sha256, speed/ETA,
//! pause/cancel, disk-space precheck (docs/08-DOWNLOAD-STORAGE.md §8.2).

pub mod manager;

pub use manager::{format_progress_line, verify_sha256_bytes, DownloadManager, DownloadProgress};
