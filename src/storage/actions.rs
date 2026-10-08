//! Storage actions (docs/08 §8.3): delete model, clear cache, move models /
//! change root, plus content-hash dedup ref-count helpers on top of
//! [`super::layout::dedup_path`].
//!
//! Conventions:
//! * one model lives at `models/<model_id>/` with rows in `models` +
//!   `model_files` (`model_files.dedup_hash` is the ref-count source —
//!   ref-counts are *derived* by counting rows, never stored separately, so
//!   they cannot drift);
//! * shared blobs live once at `cache/blobs/<sha256>` (docs/08 §8.3:
//!   `Model A+B -> shared VAE`).
//! * the [`sqlx::SqlitePool`] is passed in — same pool pattern as
//!   [`crate::model::ModelManager`]; nothing here opens its own connection.

use super::layout::{dedup_path, StorageLayout};
use crate::core::Result;
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};

/// Per-bucket byte totals for the Settings → Storage breakdown
/// (`Models 82GB / Runtimes 14GB / Caches 9GB / …`, docs/08 §8.3).
#[derive(Debug, Clone, Default)]
pub struct StorageUsage {
    pub models_bytes: u64,
    pub runtimes_bytes: u64,
    pub environments_bytes: u64,
    pub cache_bytes: u64,
    pub outputs_bytes: u64,
    pub logs_bytes: u64,
    pub plugins_bytes: u64,
}

impl StorageUsage {
    pub fn total_bytes(&self) -> u64 {
        self.models_bytes
            + self.runtimes_bytes
            + self.environments_bytes
            + self.cache_bytes
            + self.outputs_bytes
            + self.logs_bytes
            + self.plugins_bytes
    }
}

/// What [`delete_model`] removed.
#[derive(Debug, Clone, Default)]
pub struct DeleteReport {
    pub model_id: String,
    pub files_removed: u64,
    pub bytes_freed: u64,
    /// Shared blobs whose ref-count dropped to zero and were deleted.
    pub blobs_pruned: u64,
}

/// What [`clear_cache`] removed.
#[derive(Debug, Clone, Default)]
pub struct ClearReport {
    pub files_removed: u64,
    pub bytes_freed: u64,
}

/// Actions bound to one relocatable [`StorageLayout`] root.
#[derive(Debug, Clone)]
pub struct StorageActions {
    pub layout: StorageLayout,
}

impl StorageActions {
    pub fn new(layout: StorageLayout) -> Self {
        Self { layout }
    }

    /// Byte breakdown across all well-known subdirs (missing dirs count 0).
    pub fn disk_usage(&self) -> StorageUsage {
        StorageUsage {
            models_bytes: dir_size(&self.layout.models()),
            runtimes_bytes: dir_size(&self.layout.runtimes()),
            environments_bytes: dir_size(&self.layout.environments()),
            cache_bytes: dir_size(&self.layout.cache()),
            outputs_bytes: dir_size(&self.layout.outputs()),
            logs_bytes: dir_size(&self.layout.logs()),
            plugins_bytes: dir_size(&self.layout.plugins()),
        }
    }

    /// Change the data root: create the full tree at `new_root`, point this
    /// layout at it, and return the old root. Byte migration is caller-driven
    /// via [`relocate_dir`] (rename when on one drive, copy+delete across
    /// drives); until the caller moves bytes the new root is simply empty.
    pub fn change_root(&mut self, new_root: PathBuf) -> std::io::Result<PathBuf> {
        let new_layout = StorageLayout::new(new_root);
        new_layout.ensure()?;
        let old = std::mem::replace(&mut self.layout, new_layout);
        Ok(old.root)
    }

    /// Delete a model: remove `models/<id>/`, its `model_files` rows, its
    /// `models` row, then prune blobs nobody references anymore.
    pub async fn delete_model(&self, pool: &SqlitePool, model_id: &str) -> Result<DeleteReport> {
        let model_dir = self.layout.models().join(model_id);
        let bytes_freed = dir_size(&model_dir);
        let files_removed = count_files(&model_dir);
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)?;
        }
        sqlx::query("DELETE FROM model_files WHERE model_id = ?")
            .bind(model_id)
            .execute(pool)
            .await?;
        sqlx::query("DELETE FROM models WHERE id = ?")
            .bind(model_id)
            .execute(pool)
            .await?;
        let blobs_pruned = prune_orphan_blobs(pool, &self.layout.cache()).await?;
        Ok(DeleteReport {
            model_id: model_id.to_string(),
            files_removed,
            bytes_freed,
            blobs_pruned,
        })
    }

    /// Clear the cache: delete every blob no model references plus stray
    /// non-blob cache files. Referenced shared blobs are kept (ref-count > 0).
    pub async fn clear_cache(&self, pool: &SqlitePool) -> Result<ClearReport> {
        let mut report = ClearReport::default();
        let blobs = self.layout.cache().join("blobs");
        if blobs.is_dir() {
            for entry in std::fs::read_dir(&blobs)? {
                let entry = entry?;
                if !entry.file_type()?.is_file() {
                    continue;
                }
                let hash = entry.file_name().to_string_lossy().into_owned();
                if refcount(pool, &hash).await? == 0 {
                    report.bytes_freed += entry.metadata().map(|m| m.len()).unwrap_or(0);
                    std::fs::remove_file(entry.path())?;
                    report.files_removed += 1;
                }
            }
        }
        // Stray partials / temp files directly under cache/ (never blobs).
        if self.layout.cache().is_dir() {
            for entry in std::fs::read_dir(self.layout.cache())? {
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if entry.file_type()?.is_file()
                    && (name.ends_with(".part") || name.ends_with(".tmp"))
                {
                    report.bytes_freed += entry.metadata().map(|m| m.len()).unwrap_or(0);
                    std::fs::remove_file(entry.path())?;
                    report.files_removed += 1;
                }
            }
        }
        Ok(report)
    }

    /// Store `bytes` under `cache/blobs/<hash>` (idempotent — existing blob
    /// with the same hash is kept as-is). Callers insert the matching
    /// `model_files` row separately; the blob file itself carries no count.
    pub fn store_blob(&self, hash_hex: &str, bytes: &[u8]) -> Result<PathBuf> {
        let dest = dedup_path(&self.layout.cache(), hash_hex);
        if !dest.is_file() {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&dest, bytes)?;
        }
        Ok(dest)
    }

    /// Delete the blob file only when no `model_files` row references it.
    /// Returns `true` when the file was actually removed.
    pub async fn release_blob(&self, pool: &SqlitePool, hash_hex: &str) -> Result<bool> {
        if refcount(pool, hash_hex).await? > 0 {
            return Ok(false);
        }
        let dest = dedup_path(&self.layout.cache(), hash_hex);
        if dest.is_file() {
            std::fs::remove_file(dest)?;
            return Ok(true);
        }
        Ok(false)
    }
}

/// How many `model_files` rows reference a content hash — the ref-count
/// (derived from rows, so it cannot drift from a stored counter).
pub async fn refcount(pool: &SqlitePool, hash_hex: &str) -> Result<u64> {
    let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM model_files WHERE dedup_hash = ?")
        .bind(hash_hex)
        .fetch_one(pool)
        .await?;
    Ok(n.max(0) as u64)
}

/// Delete every blob file with ref-count zero. Returns blobs pruned.
pub async fn prune_orphan_blobs(pool: &SqlitePool, cache_dir: &Path) -> Result<u64> {
    let blobs = cache_dir.join("blobs");
    if !blobs.is_dir() {
        return Ok(0);
    }
    let live: Vec<String> =
        sqlx::query_scalar("SELECT DISTINCT dedup_hash FROM model_files WHERE dedup_hash IS NOT NULL")
            .fetch_all(pool)
            .await?;
    let live: std::collections::HashSet<&str> = live.iter().map(|s| s.as_str()).collect();
    let mut pruned = 0u64;
    for entry in std::fs::read_dir(&blobs)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !live.contains(name.as_str()) {
            std::fs::remove_file(entry.path())?;
            pruned += 1;
        }
    }
    Ok(pruned)
}

/// Move a directory tree to a new location (used for root/model migration):
/// fast rename on one drive, recursive copy+delete across drives.
pub fn relocate_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    if src == dst {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if std::fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    copy_tree(src, dst)?;
    std::fs::remove_dir_all(src)?;
    Ok(())
}

fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), to)?;
        }
    }
    Ok(())
}

fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

fn count_files(dir: &Path) -> u64 {
    let mut n = 0u64;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                n += 1;
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::layout::content_hash_hex;

    fn test_layout(name: &str) -> StorageLayout {
        let root = std::env::temp_dir().join(format!("nexora-storage-test-{name}"));
        let _ = std::fs::remove_dir_all(&root);
        let layout = StorageLayout::new(root);
        layout.ensure().unwrap();
        layout
    }

    async fn mem_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        let schema = include_str!("../../migrations/001_init.sql");
        for stmt in schema.split(';') {
            let stmt = stmt.trim();
            if !stmt.is_empty() {
                sqlx::query(stmt).execute(&pool).await.unwrap();
            }
        }
        pool
    }

    #[tokio::test]
    async fn shared_blob_survives_single_model_delete() {
        let layout = test_layout("shared");
        let pool = mem_pool().await;
        let actions = StorageActions::new(layout.clone());
        // Shared VAE blob referenced by models A and B (docs/08 §8.3).
        let hash = content_hash_hex(b"shared-vae");
        actions.store_blob(&hash, b"shared-vae").unwrap();
        for model in ["model-a", "model-b"] {
            let dir = layout.models().join(model);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("weights.bin"), b"weights").unwrap();
            sqlx::query("INSERT INTO models(id, name, repository, status) VALUES(?,?,?,?)")
                .bind(model)
                .bind(model)
                .bind("owner/repo")
                .bind("INSTALLED")
                .execute(&pool)
                .await
                .unwrap();
            sqlx::query("INSERT INTO model_files(model_id, path, dedup_hash) VALUES(?,?,?)")
                .bind(model)
                .bind("vae.safetensors")
                .bind(&hash)
                .execute(&pool)
                .await
                .unwrap();
        }
        assert_eq!(refcount(&pool, &hash).await.unwrap(), 2);

        let report = actions.delete_model(&pool, "model-a").await.unwrap();
        assert_eq!(report.model_id, "model-a");
        assert!(report.bytes_freed > 0);
        assert!(!layout.models().join("model-a").exists());
        // B still references the blob — file must survive.
        assert_eq!(refcount(&pool, &hash).await.unwrap(), 1);
        assert!(dedup_path(&layout.cache(), &hash).is_file());
        assert_eq!(report.blobs_pruned, 0);

        // Deleting B orphans the blob — released on prune.
        actions.delete_model(&pool, "model-b").await.unwrap();
        assert!(!dedup_path(&layout.cache(), &hash).is_file());
    }

    #[tokio::test]
    async fn clear_cache_keeps_referenced_blobs() {
        let layout = test_layout("cache");
        let pool = mem_pool().await;
        let actions = StorageActions::new(layout.clone());
        let live = content_hash_hex(b"live");
        let dead = content_hash_hex(b"dead");
        actions.store_blob(&live, b"live").unwrap();
        actions.store_blob(&dead, b"dead").unwrap();
        std::fs::write(layout.cache().join("stale.tmp"), b"tmp").unwrap();
        sqlx::query("INSERT INTO model_files(model_id, path, dedup_hash) VALUES(?,?,?)")
            .bind("m")
            .bind("f")
            .bind(&live)
            .execute(&pool)
            .await
            .unwrap();
        let report = actions.clear_cache(&pool).await.unwrap();
        assert_eq!(report.files_removed, 2); // dead blob + stale.tmp
        assert!(dedup_path(&layout.cache(), &live).is_file());
        assert!(!dedup_path(&layout.cache(), &dead).is_file());
        let _ = std::fs::remove_dir_all(&layout.root);
    }

    #[test]
    fn usage_and_root_relocation() {
        let layout = test_layout("usage");
        std::fs::write(layout.models().join("w.bin"), vec![0u8; 1024]).unwrap();
        let actions = StorageActions::new(layout);
        let usage = actions.disk_usage();
        assert_eq!(usage.models_bytes, 1024);
        assert_eq!(usage.total_bytes(), 1024);

        let mut actions = actions;
        let new_root = std::env::temp_dir().join("nexora-storage-test-usage-new");
        let _ = std::fs::remove_dir_all(&new_root);
        let old = actions.change_root(new_root.clone()).unwrap();
        assert!(new_root.join("models").is_dir());
        assert!(new_root.join("cache").join("blobs").is_dir());
        // Migrate bytes over, then usage reads from the new root.
        relocate_dir(&old.join("models"), &actions.layout.models()).unwrap();
        assert_eq!(actions.disk_usage().models_bytes, 1024);
        let _ = std::fs::remove_dir_all(&old);
        let _ = std::fs::remove_dir_all(&new_root);
    }
}
