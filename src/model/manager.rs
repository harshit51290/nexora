//! Model lifecycle orchestration. State transitions are validated with
//! [`crate::core::ModelState::can_transition`] and persisted to `models`.

use crate::core::{ModelState, NexoraError, Result};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

/// Row metadata mirrored from `models` (subset used by the UI/services).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub name: String,
    pub repository: String,
    pub revision: Option<String>,
    pub task: Option<String>,
    pub runtime: Option<String>,
    pub size_bytes: Option<i64>,
    pub status: ModelState,
    pub capabilities: Vec<String>,
    pub license: Option<String>,
    pub trust_level: Option<String>,
    /// hf-mem port results (`src/mem`, docs/05 §5.8). NULL until estimated;
    /// gates fall back to the `size_bytes × 1.2` heuristic when absent.
    pub est_weights_bytes: Option<i64>,
    pub est_kv_bytes: Option<i64>,
    pub est_total_bytes: Option<i64>,
}

/// Owns model lifecycle transitions. Inference itself is delegated to
/// child-process runtimes — never executed here (orchestration-only).
#[derive(Debug, Clone)]
pub struct ModelManager {
    pool: SqlitePool,
}

impl ModelManager {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Insert a DISCOVERED row for a repo URL (import entry point).
    pub async fn discover(&self, id: &str, name: &str, repository: &str) -> Result<ModelRecord> {
        let rec = ModelRecord {
            id: id.to_string(),
            name: name.to_string(),
            repository: repository.to_string(),
            revision: None,
            task: None,
            runtime: None,
            size_bytes: None,
            status: ModelState::Discovered,
            capabilities: vec![],
            license: None,
            trust_level: None,
            est_weights_bytes: None,
            est_kv_bytes: None,
            est_total_bytes: None,
        };
        self.insert_row(&rec).await?;
        Ok(rec)
    }

    pub async fn metadata(&self, id: &str) -> Result<Option<ModelRecord>> {
        use sqlx::Row;
        let row = sqlx::query(
            "SELECT id, name, repository, revision, task, runtime, size_bytes, status, capabilities, license, trust_level, est_weights_bytes, est_kv_bytes, est_total_bytes FROM models WHERE id = ?",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|r| {
            let status: String = r.get("status");
            let capabilities: String = r.get("capabilities");
            ModelRecord {
                id: r.get("id"),
                name: r.get("name"),
                repository: r.get("repository"),
                revision: r.get("revision"),
                task: r.get("task"),
                runtime: r.get("runtime"),
                size_bytes: r.get("size_bytes"),
                status: status.parse().unwrap_or(ModelState::Error),
                capabilities: serde_json::from_str(&capabilities).unwrap_or_default(),
                license: r.get("license"),
                trust_level: r.get("trust_level"),
                est_weights_bytes: r.get("est_weights_bytes"),
                est_kv_bytes: r.get("est_kv_bytes"),
                est_total_bytes: r.get("est_total_bytes"),
            }
        }))
    }

    /// Validated + persisted transition helper (used by install/load/unload).
    pub async fn transition(&self, id: &str, next: ModelState) -> Result<()> {
        let current = self
            .metadata(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("unknown model {id}"))?
            .status;
        if !current.can_transition(next) {
            return Err(anyhow::anyhow!("illegal model transition {current} -> {next}").into());
        }
        sqlx::query("UPDATE models SET status = ? WHERE id = ?")
            .bind(next.as_str())
            .bind(id)
            .execute(&self.pool)
            .await?;
        tracing::info!(%id, %current, %next, "model transition");
        Ok(())
    }

    pub async fn install(&self, id: &str) -> Result<()> {
        // Full 8-step flow (dir, env, download, verify, deps, validate,
        // test-load, library add) is driven by services; the manager only
        // advances the persisted SUPPORTED -> DOWNLOADING leg here.
        self.transition(id, ModelState::Downloading).await
    }

    pub async fn mark_installed(&self, id: &str) -> Result<()> {
        self.transition(id, ModelState::Installed).await
    }

    pub async fn load(&self, id: &str) -> Result<()> {
        // VRAM gate + compat estimate must pass BEFORE this is called
        // (AGENTS.md rule 6; see hardware backend + analyzer).
        let rec = self
            .metadata(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("unknown model {id}"))?;
        if !matches!(rec.status, ModelState::Ready | ModelState::Loaded) {
            return Err(NexoraError::Other(anyhow::anyhow!(
                "model {id} must be READY before load (is {})",
                rec.status
            )));
        }
        let next = if rec.status == ModelState::Ready {
            ModelState::Loaded
        } else {
            ModelState::Running
        };
        self.transition(id, next).await
    }

    pub async fn unload(&self, id: &str) -> Result<()> {
        self.transition(id, ModelState::Ready).await
    }

    pub async fn remove(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM models WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn insert_row(&self, rec: &ModelRecord) -> Result<()> {
        sqlx::query(
            "INSERT INTO models(id, name, repository, revision, task, runtime, size_bytes, status, capabilities, license, trust_level) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        )
        .bind(&rec.id)
        .bind(&rec.name)
        .bind(&rec.repository)
        .bind(&rec.revision)
        .bind(&rec.task)
        .bind(&rec.runtime)
        .bind(rec.size_bytes)
        .bind(rec.status.as_str())
        .bind(serde_json::to_string(&rec.capabilities).unwrap_or_else(|_| "[]".into()))
        .bind(&rec.license)
        .bind(&rec.trust_level)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
