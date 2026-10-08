//! Core service wiring (`docs/10-API-CLI.md` §10.1–10.2, `docs/04` §4.1–4.2).
//!
//! Single bootstrap point shared by REST, OpenAI-compat, WS and the CLI.
//! Every function below calls a REAL manager and persists state:
//!
//! * models → [`ModelManager`] (every [`ModelState`] transition persisted)
//! * hardware → [`HardwareBackend`] (`NvidiaBackend` over a CPU baseline)
//! * downloads → [`DownloadManager`] (resume/pause/cancel/checksum/space)
//! * generate → [`Scheduler`] admission + `RuntimeAdapter::prepare` → `run`
//!   (AGENTS.md rule 3), VRAM gate BEFORE load (rule 6), generation row +
//!   output sidecar persisted on success (docs/04 §4.1, §4.5)
//! * lists → direct SQLx queries over `models` / `runtimes`
//!
//! Data root: `NEXORA_DATA_DIR` env or the relocatable default
//! (`StorageLayout::default_root`). The SQLite schema is applied by the
//! versioned runner in [`super::migrations`] (records in `_schema_migrations`
//! — the schema no longer replays `001` by hand; NEED-CARGO: `sqlx::migrate!`
//! needs the `sqlx` `"migrate"` feature, which `Cargo.toml` does not enable).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use sqlx::{Row, SqlitePool};
use tokio::io::AsyncReadExt;

use crate::analyze;
use crate::core::ModelState;
use crate::download::DownloadManager;
use crate::hardware::{
    select_execution_mode, vram_fits, CpuBackend, ExecutionMode, HardwareBackend, NvidiaBackend,
};
use crate::hf::{fetch_metadata, HfRepo};
use crate::jobs::JobQueue;
use crate::model::manager::ModelRecord;
use crate::model::ModelManager;
use crate::runtime::{self, InferenceRequest, RuntimeAdapter};
use crate::scheduler::{Job, JobState, Scheduler};
use crate::security::classify_trust;
use crate::storage::layout::{OutputKind, StorageLayout};

use super::core_stub::{self, CoreStubError};
use super::migrations::apply_migrations;
use super::versions::adapter_version;
use super::{
    ExecutionPrefs, GenerateRequest, GenerationResult, GenerationSidecar, HardwareSummary,
    ModelSummary, RuntimeSummary,
};

/// MVP adapter ids surfaced when the `runtimes` table has no row yet
/// (docs/03 §3.1). Status for those rows is a live env probe, not stored;
/// version comes from [`adapter_version`] (bundle stamp or bundled crate
/// version — never `"unknown"`).
const KNOWN_RUNTIMES: &[(&str, &str)] =
    &[("transformers", "transformers"), ("diffusers", "diffusers"), ("llama_cpp", "llama.cpp")];

// ---------------------------------------------------------------------------
// Handle + bootstrap
// ---------------------------------------------------------------------------

/// Cloned into every handler and background install task.
#[derive(Debug, Clone)]
pub(super) struct CoreHandle {
    pool: SqlitePool,
    data_root: PathBuf,
    models: ModelManager,
    downloads: DownloadManager,
    scheduler: Arc<tokio::sync::Mutex<Scheduler>>,
}

/// Process-wide handle (async-built once; cheap clones after).
static CORE: tokio::sync::Mutex<Option<CoreHandle>> = tokio::sync::Mutex::const_new(None);

pub(super) fn data_root() -> PathBuf {
    std::env::var("NEXORA_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| StorageLayout::default_root().root)
}

pub(super) async fn core() -> Result<CoreHandle, CoreStubError> {
    {
        let guard = CORE.lock().await;
        if let Some(handle) = guard.as_ref() {
            return Ok(handle.clone());
        }
    }
    let fresh = CoreHandle::bootstrap(data_root()).await?;
    let mut guard = CORE.lock().await;
    if let Some(handle) = guard.as_ref() {
        return Ok(handle.clone());
    }
    *guard = Some(fresh.clone());
    Ok(fresh)
}

impl CoreHandle {
    pub(super) async fn bootstrap(data_root: PathBuf) -> Result<Self, CoreStubError> {
        let layout = StorageLayout::new(data_root.clone());
        layout.ensure()?;
        let db_path = data_root.join("nexora.db");
        let opts = sqlx::sqlite::SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(true);
        let pool = SqlitePool::connect_with(opts).await?;
        apply_schema(&pool).await?;
        // VRAM budget = currently free VRAM when a GPU exists; `None` on
        // CPU-only machines (CPU path always admits — slowly).
        let budget = NvidiaBackend::memory().vram_free_mb;
        Ok(Self {
            models: ModelManager::new(pool.clone()),
            pool,
            data_root,
            downloads: DownloadManager::default(),
            scheduler: Arc::new(tokio::sync::Mutex::new(Scheduler::new(1, budget))),
        })
    }

    fn model_dir(&self, model_id: &str) -> PathBuf {
        StorageLayout::new(self.data_root.clone())
            .models()
            .join(model_id.replace(['/', '\\'], "__"))
    }

    // -- models -----------------------------------------------------------

    pub(super) async fn model_record(&self, id: &str) -> Result<ModelRecord, CoreStubError> {
        self.models.metadata(id).await?.ok_or_else(|| {
            CoreStubError::coded(
                "E_MODEL_NOT_FOUND",
                format!("unknown model {id}"),
                "Install it first: `uar install <owner/model>` or POST /models/install.",
            )
        })
    }

    pub(super) async fn install_model(
        &self,
        repo: &str,
        revision: Option<&str>,
    ) -> Result<String, CoreStubError> {
        // Canonicalize (bare `owner/model` passes through unchanged).
        let repo = core_stub::parse_hf_url(repo)?;
        let rev = revision.unwrap_or("main").to_string();
        let model_id = repo.clone();
        let name = repo.rsplit('/').next().unwrap_or(&repo).to_string();

        if self.models.metadata(&model_id).await?.is_none() {
            self.models.discover(&model_id, &name, &repo).await?;
        }
        // Walk forward to SUPPORTED; rows already past it (re-install)
        // keep their status and go straight to re-fetch.
        for next in [ModelState::Analyzing, ModelState::Supported] {
            let cur = self.model_record(&model_id).await?.status;
            if cur == next {
                continue;
            }
            if cur.can_transition(next) {
                self.models.transition(&model_id, next).await?;
            } else {
                break;
            }
        }
        // Analyzer pre-classification (docs/12 B3 gate). NEED: full
        // analyzer (`registry.json` compat score, docs/05 §5.5).
        let rec = core_stub::classify_model(&repo);
        let caps: Vec<String> = if rec.task == "unknown" {
            vec![]
        } else {
            vec![rec.task.clone()]
        };
        sqlx::query("UPDATE models SET task = ?, runtime = ?, capabilities = ? WHERE id = ?")
            .bind(&rec.task)
            .bind(&rec.runtime)
            .bind(serde_json::to_string(&caps).unwrap_or_else(|_| "[]".into()))
            .bind(&model_id)
            .execute(&self.pool)
            .await?;
        // Synchronously enter DOWNLOADING so list views reflect it; the
        // bytes move in a background task.
        if self.model_record(&model_id).await?.status == ModelState::Supported {
            self.models.install(&model_id).await?;
        }
        let me = self.clone();
        let (job_id, job_repo, job_rev) = (model_id.clone(), repo, rev);
        tokio::spawn(async move {
            if let Err(e) = me.install_job(&job_id, &job_repo, &job_rev).await {
                if e.message.contains("download cancelled") {
                    // User-cancelled: stay resumable, don't ERROR the model.
                    let _ = sqlx::query(
                        "UPDATE downloads SET state = 'Cancelled' WHERE model_id = ?",
                    )
                    .bind(&job_id)
                    .execute(&me.pool)
                    .await;
                    tracing::info!(model = %job_id, "install cancelled by user (resumable)");
                } else {
                    tracing::error!(model = %job_id, code = e.code, msg = %e.message, "install job failed");
                    let _ = me.mark_error(&job_id).await;
                }
            }
        });
        Ok(model_id)
    }

    /// Background half of install: metadata → space/VRAM prechecks →
    /// per-file resume downloads → VALIDATING → READY.
    async fn install_job(&self, model_id: &str, repo: &str, rev: &str) -> Result<(), CoreStubError> {
        if self.model_record(model_id).await?.status == ModelState::Supported {
            self.models.install(model_id).await?;
        }
        let mut parts = repo.split('/');
        let hf = HfRepo {
            owner: parts.next().unwrap_or("").to_string(),
            repo: parts.next().unwrap_or("").to_string(),
            rev: rev.to_string(),
        };
        let meta = fetch_metadata(&hf).await?;
        let dir = self.model_dir(model_id);
        tokio::fs::create_dir_all(&dir).await?;

        // Metadata-only trust/license signals (non-interactive: record the
        // level; the View/Sandbox/Cancel modal lives in the desktop UI —
        // see deviation note on `gate_custom_code`).
        let files = meta.file_names();
        let trust = classify_trust(false, files.iter().any(|f| f.ends_with(".py")), false);
        sqlx::query("UPDATE models SET license = ?, trust_level = ? WHERE id = ?")
            .bind(meta.license())
            .bind(trust.as_str())
            .bind(model_id)
            .execute(&self.pool)
            .await?;

        // Disk-space precheck BEFORE the first byte (E_SPACE_LOW).
        let total: u64 = meta.siblings.iter().filter_map(|s| s.size).sum();
        if total > 0 {
            self.downloads.precheck_space(&dir, total).await?;
        } else {
            tracing::warn!(model = %model_id, "Hub reported no file sizes; skipping space precheck");
        }
        // hf-mem port (docs/05 §5.8): REAL weight/KV bytes BEFORE the bytes
        // move. Never fails the install — any error falls back to Hub sizes.
        let mut chosen_gguf: Option<String> = None;
        let mut real_total: Option<u64> = None;
        match crate::mem::estimate_repo(
            &hf,
            &crate::mem::EstimateOpts {
                experimental: true,
                ..Default::default()
            },
        )
        .await
        {
            Ok(est) => {
                sqlx::query(
                    "UPDATE models SET est_weights_bytes = ?, est_kv_bytes = ?, est_total_bytes = ?, est_at = ? WHERE id = ?",
                )
                .bind(est.weights_bytes as i64)
                .bind(est.kv_bytes.map(|b| b as i64))
                .bind(est.total_bytes.map(|b| b as i64))
                .bind(chrono::Utc::now().to_rfc3339())
                .bind(model_id)
                .execute(&self.pool)
                .await?;
                real_total = est.total_bytes.or(Some(est.weights_bytes));
                // Multi-GGUF repos: download the recommended variant only
                // (+ tokenizer/config sidecars), not every quant.
                let ggufs: Vec<_> = est
                    .per_file
                    .iter()
                    .filter(|f| f.name.ends_with(".gguf"))
                    .collect();
                if ggufs.len() > 1 {
                    let files: Vec<crate::mem::FileEstimate> =
                        ggufs.into_iter().cloned().collect();
                    let free = NvidiaBackend::memory()
                        .vram_free_mb
                        .or(NvidiaBackend::memory().vram_total_mb);
                    match crate::mem::pick_gguf_variant(&files, free) {
                        Some(name) => {
                            tracing::info!(model = %model_id, variant = %name, "GGUF variant picked for this GPU; other quants skipped");
                            chosen_gguf = Some(name);
                        }
                        None => {
                            let smallest =
                                crate::mem::smallest_variant(&files).unwrap_or_default();
                            tracing::warn!(model = %model_id, fallback = %smallest,
                                "no GGUF variant fits VRAM; downloading smallest with CPU-offload note");
                            chosen_gguf = Some(smallest);
                        }
                    }
                }
            }
            Err(e) => tracing::warn!(model = %model_id, error = %e,
                "memory estimate failed; install continues on Hub sizes"),
        }
        // Compat estimate BEFORE the bytes move (docs/12 C5): real measured
        // bytes when available, Hub total otherwise. Poor warns, the
        // CPU-offload path stays open — warn, don't block.
        {
            let cpu = CpuBackend::memory();
            let vram = NvidiaBackend::memory();
            let free = vram.vram_free_mb.or(vram.vram_total_mb);
            let (score, cfg) =
                analyze::compat_score(real_total.unwrap_or(total), cpu.ram_total_mb, free, false);
            if score.verdict == "Poor" {
                tracing::warn!(model = %model_id, overall = score.overall, reason = %cfg.reason,
                    "compat Poor: download continues; expect quant/offload needs");
            }
        }

        sqlx::query(
            "INSERT INTO downloads(model_id, bytes_done, bytes_total, state) VALUES(?,?,?,'Running')
             ON CONFLICT(model_id) DO UPDATE SET bytes_done = 0, bytes_total = excluded.bytes_total, state = 'Running'",
        )
        .bind(model_id)
        .bind(0i64)
        .bind(total as i64)
        .execute(&self.pool)
        .await?;

        let mut done: u64 = 0;
        for entry in &meta.siblings {
            // Multi-GGUF repos: only the picked variant downloads — the
            // rest are skipped (tokenizer/config sidecars always fetch).
            if entry.rfilename.ends_with(".gguf") {
                if let Some(chosen) = chosen_gguf.as_deref() {
                    if entry.rfilename != chosen {
                        tracing::info!(model = %model_id, skipped = %entry.rfilename,
                            "skipping non-recommended GGUF variant (re-run estimate to change)");
                        continue;
                    }
                }
            }
            let url = format!("https://huggingface.co/{repo}/resolve/{rev}/{}", entry.rfilename);
            let dest = dir.join(&entry.rfilename);
            if let Some(parent) = dest.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            // Resume + atomic rename are the transport story (`HfFileEntry`
            // carries no expected hash); integrity is recorded below.
            self.downloads.download_url(&url, &dest, None).await?;
            let bytes = tokio::fs::metadata(&dest).await.map(|m| m.len()).unwrap_or(0);
            done += entry.size.unwrap_or(bytes);
            // Per-file SHA plumbing (docs/04 §4.1 `model_files`): hash the
            // completed file, store it as both `sha256` and `dedup_hash`
            // (content-addressed dedup key, docs/08 §8.3).
            let sha_hex = sha256_file(&dest).await?;
            sqlx::query(
                "INSERT INTO model_files(model_id, path, bytes, sha256, dedup_hash)
                 VALUES(?,?,?,?,?)",
            )
            .bind(model_id)
            .bind(&entry.rfilename)
            .bind(bytes as i64)
            .bind(&sha_hex)
            .bind(&sha_hex)
            .execute(&self.pool)
            .await?;
            sqlx::query("UPDATE downloads SET bytes_done = ? WHERE model_id = ?")
                .bind(done as i64)
                .bind(model_id)
                .execute(&self.pool)
                .await?;
        }
        sqlx::query("UPDATE downloads SET state = 'Completed' WHERE model_id = ?")
            .bind(model_id)
            .execute(&self.pool)
            .await?;

        let entries = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        if entries == 0 {
            return Err(CoreStubError::coded(
                "E_INSTALL_FAILED",
                format!("model dir {} is empty after download", dir.display()),
                "Retry install; if it repeats, the Hub listing changed — report the repo URL.",
            ));
        }
        // Fresh installs walk DOWNLOADING -> INSTALLED -> VALIDATING ->
        // READY. A re-fetch over a model that already moved past
        // DOWNLOADING (re-install of READY) refreshes files in place and
        // keeps its status — there is no legal backwards leg.
        let post = self.model_record(model_id).await?.status;
        if post == ModelState::Downloading {
            self.models.mark_installed(model_id).await?;
            self.models.transition(model_id, ModelState::Validating).await?;
            self.models.transition(model_id, ModelState::Ready).await?;
            tracing::info!(model = %model_id, files = entries, "install complete -> READY");
        } else {
            tracing::info!(model = %model_id, status = %post, files = entries, "re-fetch complete; status unchanged");
        }
        Ok(())
    }

    async fn mark_error(&self, model_id: &str) -> Result<(), CoreStubError> {
        if let Some(rec) = self.models.metadata(model_id).await? {
            if rec.status.can_transition(ModelState::Error) {
                self.models.transition(model_id, ModelState::Error).await?;
            }
        }
        Ok(())
    }

    pub(super) async fn load_model(&self, model_id: &str) -> Result<(), CoreStubError> {
        self.load_model_with(model_id, &ExecutionPrefs::default())
            .await
    }

    pub(super) async fn load_model_with(
        &self,
        model_id: &str,
        prefs: &ExecutionPrefs,
    ) -> Result<(), CoreStubError> {
        let rec = self.model_record(model_id).await?;
        if matches!(rec.status, ModelState::Loaded | ModelState::Running) {
            return Ok(()); // idempotent
        }
        if rec.status != ModelState::Ready {
            return Err(CoreStubError::coded(
                "E_MODEL_NOT_READY",
                format!("model {model_id} is {} (need READY)", rec.status),
                "Wait for install to reach READY, or reinstall if it is in ERROR.",
            ));
        }
        let dir = self.model_dir(model_id);
        if !dir.is_dir() {
            return Err(CoreStubError::coded(
                "E_MODEL_NOT_FOUND",
                format!("model files missing: {}", dir.display()),
                "Reinstall the model: `uar install <owner/model>`.",
            ));
        }
        // VRAM gate BEFORE load (AGENTS.md rule 6); `--cpu`/`--offload`
        // widen it instead of hard-failing (see `gate_vram`). Measured
        // hf-mem total preferred over the size heuristic (docs/05 §5.8).
        gate_vram(est_vram_mb(&rec), prefs)?;
        // Env/runtime readiness probe (E_ENV_MISSING / E_RUNTIME_NOT_READY).
        let (adapter, _) = self.resolve_adapter(&rec)?;
        adapter.health_check()?;
        // No-op unless a stale server child lingers from a previous run.
        let _ = adapter.stop();
        self.models.load(model_id).await?;
        Ok(())
    }

    pub(super) async fn unload_model(&self, model_id: &str) -> Result<(), CoreStubError> {
        let rec = self.model_record(model_id).await?;
        if rec.status == ModelState::Ready {
            return Ok(()); // idempotent
        }
        // Best-effort runtime stop; unload proceeds regardless.
        if let Ok((adapter, _)) = self.resolve_adapter(&rec)
        {
            if let Err(e) = adapter.stop() {
                tracing::warn!(model = %model_id, code = e.code, "runtime stop during unload: {}", e.message);
            }
        }
        self.models.unload(model_id).await?;
        Ok(())
    }

    // -- downloads --------------------------------------------------------

    pub(super) async fn download_control(&self, action: &str) -> Result<String, CoreStubError> {
        let state = match action {
            "pause" => {
                self.downloads.pause();
                "Paused"
            }
            "resume" => {
                self.downloads.resume();
                "Running"
            }
            "cancel" => {
                self.downloads.cancel();
                "Cancelled"
            }
            _ => {
                return Err(CoreStubError::coded(
                    "E_BAD_DOWNLOAD_ACTION",
                    format!("unknown download action {action}"),
                    "Use pause, resume or cancel.",
                ));
            }
        };
        // Mirror into the row; missing row (no download yet) is not fatal.
        let _ = sqlx::query("UPDATE downloads SET state = ?")
            .bind(state)
            .execute(&self.pool)
            .await;
        Ok(state.to_string())
    }

    // -- lists --------------------------------------------------------------

    pub(super) async fn list_models(&self) -> Result<Vec<ModelSummary>, CoreStubError> {
        let rows = sqlx::query(
            "SELECT id, name, repository, status, capabilities FROM models ORDER BY name",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let caps: String = r.get("capabilities");
                ModelSummary {
                    id: r.get("id"),
                    name: r.get("name"),
                    repository: r.get("repository"),
                    status: r.get("status"),
                    capabilities: serde_json::from_str(&caps).unwrap_or_default(),
                }
            })
            .collect())
    }

    /// Persisted queue rows (DB is truth; docs/10 §10.4). Used by `uar jobs`.
    pub(super) async fn list_jobs(&self) -> Result<Vec<crate::jobs::Job>, CoreStubError> {
        JobQueue::list_persisted(&self.pool)
            .await
            .map_err(|e| CoreStubError::coded("E_DB", format!("jobs query failed: {e}"), "Check Logs; the data dir may be unwritable."))
    }

    pub(super) async fn hardware_info(&self) -> Result<HardwareSummary, CoreStubError> {        // NVIDIA over a CPU baseline; CPU-only machines get CPU/RAM/OS.
        let info = NvidiaBackend::detect();
        let label = info
            .gpu_label
            .clone()
            .unwrap_or_else(|| info.cpu_label.clone());
        Ok(HardwareSummary {
            label,
            gpu_vram_gb: info.vram_total_mb.map(|mb| mb as f32 / 1024.0),
            system_ram_gb: Some(info.system_ram_mb as f32 / 1024.0),
            notes: format!(
                "cpu={} ram_mb={} cuda={} driver={} os={}/{} disk_free={}",
                info.cpu_label,
                info.system_ram_mb,
                info.cuda_available,
                info.driver_version.as_deref().unwrap_or("n/a"),
                info.os,
                info.arch,
                info.disk_free_bytes,
            ),
        })
    }

    pub(super) async fn list_runtimes(&self) -> Result<Vec<RuntimeSummary>, CoreStubError> {
        let rows =
            sqlx::query("SELECT id, kind, version, status FROM runtimes ORDER BY id")
                .fetch_all(&self.pool)
                .await?;
        let mut out: Vec<RuntimeSummary> = rows
            .into_iter()
            .map(|r| {
                let id: String = r.get("id");
                let stored: Option<String> = r.get("version");
                // Rows written before versions existed carry NULL/"": resolve
                // for display rather than persisting over stored data.
                let version = match stored {
                    Some(v) if !v.trim().is_empty() => v,
                    _ => adapter_version(&self.data_root, &id),
                };
                RuntimeSummary {
                    id,
                    kind: r.get("kind"),
                    version,
                    status: r.get("status"),
                }
            })
            .collect();
        // Overlay the MVP registry for ids with no stored row (docs/03
        // §3.1). Status is a live env probe; version comes from the version
        // source (`super::versions`), never `"unknown"`.
        for &(id, kind) in KNOWN_RUNTIMES {
            if out.iter().any(|r| r.id == id) {
                continue;
            }
            let status = match runtime::adapter_for(id, self.data_root.clone()) {
                Some(a) => match a.health_check() {
                    Ok(()) => "READY",
                    Err(_) => "NOT_INSTALLED",
                },
                None => "NOT_INSTALLED",
            };
            out.push(RuntimeSummary {
                id: id.to_string(),
                kind: kind.to_string(),
                version: adapter_version(&self.data_root, id),
                status: status.to_string(),
            });
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    // -- generate -----------------------------------------------------------

    pub(super) async fn generate(
        &self,
        req: &GenerateRequest,
    ) -> Result<GenerationResult, CoreStubError> {
        if req.prompt.trim().is_empty() {
            return Err(CoreStubError::coded(
                "E_EMPTY_PROMPT",
                "Prompt is empty.".to_string(),
                "Provide a non-empty prompt string.".to_string(),
            ));
        }
        let rec = self.model_record(&req.model).await?;
        // Placement overrides ride the request so load-time and admission
        // gates agree (REST/WS/CLI share this path).
        let prefs = req.execution.clone().unwrap_or_default();
        match rec.status {
            ModelState::Ready => self.load_model_with(&req.model, &prefs).await?,
            ModelState::Loaded | ModelState::Running => {}
            _ => {
                return Err(CoreStubError::coded(
                    "E_MODEL_NOT_READY",
                    format!("model {} is {} (need READY or LOADED)", req.model, rec.status),
                    "Finish installing first, or load the model before generating.",
                ));
            }
        }

        // Scheduler admission: VRAM-gated concurrency (docs/10 §10.4).
        // `--cpu`/`--offload` widen the gate instead of hard-failing.
        // Measured hf-mem total preferred (docs/05 §5.8).
        let est_mb = est_vram_mb(&rec);
        let job_id = format!("gen-{}", uuid::Uuid::new_v4().simple());
        {
            let mut sched = self.scheduler.lock().await;
            sched.enqueue(Job {
                id: job_id.clone(),
                model_id: req.model.clone(),
                est_vram_mb: est_mb,
                state: JobState::Waiting,
            });
            match sched.start_next() {
                Ok(_) => {}
                Err(crate::core::NexoraError::VramShort { required_mb, available_mb }) => {
                    if prefs.cpu {
                        tracing::warn!(
                            job = %job_id, required_mb, available_mb,
                            "--cpu: VRAM gate bypassed; running fully on CPU (slow, high RAM use)");
                    } else {
                        match select_execution_mode(required_mb, Some(available_mb)) {
                            ExecutionMode::Offload => tracing::warn!(
                                job = %job_id, required_mb, available_mb,
                                "VRAM short: proceeding with CPU offload (slow)"),
                            // --offload widens past the normal offload window.
                            _ if prefs.offload => tracing::warn!(
                                job = %job_id, required_mb, available_mb,
                                "--offload: past the normal offload window; proceeding with heavy CPU offload (slow, RAM pressure may still OOM)"),
                            _ => {
                                sched.fail(&job_id);
                                return Err(CoreStubError::from(
                                    crate::core::NexoraError::VramShort { required_mb, available_mb },
                                ));
                            }
                        }
                    }
                }
                Err(e) => {
                    sched.fail(&job_id);
                    return Err(e.into());
                }
            }
        }

        let outcome = self.run_admitted(&rec, req).await;
        {
            let mut sched = self.scheduler.lock().await;
            match &outcome {
                Ok(_) => {
                    sched.complete(&job_id);
                }
                Err(_) => {
                    sched.fail(&job_id);
                }
            }
        }
        // Back to LOADED whatever happened (run failure != model failure).
        if let Some(r) = self.models.metadata(&req.model).await? {
            if r.status == ModelState::Running {
                let _ = self.models.transition(&req.model, ModelState::Loaded).await;
            }
        }
        outcome
    }

    async fn run_admitted(
        &self,
        rec: &ModelRecord,
        req: &GenerateRequest,
    ) -> Result<GenerationResult, CoreStubError> {
        if self.model_record(&req.model).await?.status == ModelState::Loaded {
            // Legal leg only; ignore when already Running (concurrent gen).
            let _ = self.models.transition(&req.model, ModelState::Running).await;
        }
        let (adapter, model_dir) = self.resolve_adapter(rec)?;
        // AGENTS.md rule 3: callers use prepare -> run only.
        adapter.prepare(rec, &model_dir)?;

        let mut params: HashMap<String, String> = match req.params.clone().unwrap_or_default() {
            serde_json::Value::Object(map) => map
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        match v {
                            serde_json::Value::String(s) => s,
                            other => other.to_string(),
                        },
                    )
                })
                .collect(),
            other => {
                let mut m = HashMap::new();
                m.insert("value".to_string(), other.to_string());
                m
            }
        };
        if let Some(w) = req.width {
            params.entry("width".into()).or_insert_with(|| w.to_string());
        }
        if let Some(h) = req.height {
            params.entry("height".into()).or_insert_with(|| h.to_string());
        }
        // `--gpu-layers N`: layers to keep on GPU when offloading
        // (llama.cpp-style `n_gpu_layers`; adapters ignore unknown params).
        // Explicit request params win over the flag.
        if let Some(n) = req.execution.as_ref().and_then(|e| e.gpu_layers) {
            params
                .entry("n_gpu_layers".into())
                .or_insert_with(|| n.to_string());
        }
        let out_dir = self.output_dir_for(&rec.capabilities);
        std::fs::create_dir_all(&out_dir)?;
        let infreq = InferenceRequest {
            model: rec.clone(),
            model_dir,
            prompt: req.prompt.clone(),
            negative_prompt: params.get("negative_prompt").cloned(),
            params,
            seed: req.seed,
            output_dir: Some(out_dir.clone()),
        };
        // Child-process inference can run for minutes: never block the
        // async runtime while the supervised child executes.
        let result = tokio::task::spawn_blocking(move || adapter.run(infreq))
            .await
            .map_err(|e| {
                CoreStubError::coded(
                    "E_RUNTIME_CRASH",
                    format!("inference task aborted: {e}"),
                    "Retry the generate; if it repeats, check Logs and reinstall the runtime env.",
                )
            })??;

        let gen_id = uuid::Uuid::new_v4().simple().to_string();
        let text = result.text.clone();
        let artifact_path = if !result.files.is_empty() {
            result.files[0].to_string_lossy().to_string()
        } else if let Some(t) = text.clone() {
            let path = out_dir.join(format!("gen-{gen_id}.txt"));
            tokio::fs::write(&path, t.as_bytes()).await?;
            path.to_string_lossy().to_string()
        } else {
            // Adapter returned neither files nor text: persist the sidecar
            // as the artifact so the run stays reproducible/auditable.
            let path = out_dir.join(format!("gen-{gen_id}.json"));
            path.to_string_lossy().to_string()
        };
        let params_value = req.params.clone().unwrap_or(serde_json::Value::Null);
        let sidecar = GenerationSidecar::new(
            req.model.clone(),
            req.prompt.clone(),
            req.seed,
            params_value,
            result.runtime_id.clone(),
        );
        let sidecar_json = serde_json::to_string_pretty(&sidecar).unwrap_or_else(|_| "{}".into());
        if artifact_path.ends_with(".json") {
            tokio::fs::write(&artifact_path, sidecar_json.as_bytes()).await?;
        } else {
            tokio::fs::write(format!("{artifact_path}.json"), sidecar_json.as_bytes()).await?;
        }
        sqlx::query(
            "INSERT INTO generations(id, model_id, prompt, params, seed, output_path, created_at, runtime)
             VALUES(?,?,?,?,?,?,?,?)",
        )
        .bind(&gen_id)
        .bind(&req.model)
        .bind(&req.prompt)
        .bind(req.params.clone().map(|v| v.to_string()).unwrap_or_else(|| "null".into()))
        .bind(req.seed.map(|s| s as i64))
        .bind(&artifact_path)
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(&result.runtime_id)
        .execute(&self.pool)
        .await?;
        Ok(GenerationResult {
            id: gen_id,
            output_path: artifact_path,
            text,
            sidecar,
        })
    }

    /// Stored runtime id → adapter; else first adapter whose `detect()`
    /// claims the on-disk layout; else a real unsupported-model error.
    fn resolve_adapter(
        &self,
        rec: &ModelRecord,
    ) -> Result<(Box<dyn RuntimeAdapter>, PathBuf), CoreStubError> {
        let model_id = &rec.id;
        let model_dir = self.model_dir(model_id);
        if let Some(raw) = rec.runtime.as_deref() {
            // `classify_model` predates the registry and says "llama.cpp";
            // the registry id is "llama_cpp".
            let normalized = raw.to_lowercase().replace('.', "_");
            if normalized == "unsupported" {
                return Err(CoreStubError::coded(
                    "E_MODEL_UNSUPPORTED",
                    format!("model {model_id} is classified unsupported (runtime={raw})"),
                    "Pick a supported architecture, or retry via the Custom Python adapter in Advanced mode.",
                ));
            }
            if let Some(a) = runtime::adapter_for(&normalized, self.data_root.clone()) {
                return Ok((a, model_dir));
            }
            return Err(CoreStubError::coded(
                "E_RUNTIME_NOT_FOUND",
                format!("stored runtime '{raw}' has no adapter"),
                "Reinstall the model so the analyzer refreshes its runtime mapping.",
            ));
        }
        if let Some(a) = runtime::all_adapters(self.data_root.clone())
            .into_iter()
            .find(|a| a.detect(rec, &model_dir))
        {
            return Ok((a, model_dir));
        }
        Err(CoreStubError::coded(
            "E_MODEL_UNSUPPORTED",
            format!("no runtime claims model {model_id}"),
            "Check the model page for the recommended runtime, or try the Custom Python adapter in Advanced mode.",
        ))
    }

    fn output_dir_for(&self, capabilities: &[String]) -> PathBuf {
        let layout = StorageLayout::new(self.data_root.clone());
        let has = |tok: &str| capabilities.iter().any(|c| c == tok);
        if ["text-to-image", "image-to-image", "image-editing"].iter().any(|t| has(t)) {
            layout.output_dir(OutputKind::Images)
        } else if ["text-to-speech", "speech-to-text", "audio-generation"].iter().any(|t| has(t)) {
            layout.output_dir(OutputKind::Audio)
        } else if has("video-generation") {
            layout.output_dir(OutputKind::Video)
        } else {
            layout.output_dir(OutputKind::Text)
        }
    }
}

/// VRAM estimate in MB: measured hf-mem total when present (no fudge
/// factor — it already counts weights + KV), else the legacy
/// `size_bytes × 1.2` heuristic, else a 2GB placeholder (docs/07 §7.4).
fn est_vram_mb(rec: &ModelRecord) -> u64 {
    if let Some(total) = rec.est_total_bytes {
        return (total as u64).div_ceil(1024 * 1024);
    }
    rec.size_bytes
        .map(|b| (b as u64) / 1024 / 1024 * 12 / 10)
        .unwrap_or(2048)
}

/// VRAM gate shared by load and generate paths. CPU-only machines always
/// pass; GPU machines warn-and-proceed inside the offload window and fail
/// past it (fix points at quant/offload/unload) — unless the caller widens
/// the window: `--cpu` skips the gate (slow CPU run), `--offload` proceeds
/// with heavy CPU offload even past the normal window (slow + RAM pressure).
fn gate_vram(est_mb: u64, prefs: &ExecutionPrefs) -> Result<(), CoreStubError> {
    if prefs.cpu {
        tracing::warn!(
            est_mb,
            "--cpu: skipping VRAM gate; model runs fully on CPU (slow, high RAM use)"
        );
        return Ok(());
    }
    let free = NvidiaBackend::memory().vram_free_mb;
    match vram_fits(est_mb, free) {
        Ok(()) => Ok(()),
        Err(_) => match select_execution_mode(est_mb, free) {
            ExecutionMode::Offload => {
                tracing::warn!(est_mb, free_mb = ?free, "VRAM short: proceeding with CPU offload (slow)");
                Ok(())
            }
            _ if prefs.offload => {
                tracing::warn!(est_mb, free_mb = ?free, "--offload: past the normal offload window; proceeding with heavy CPU offload (slow, RAM pressure may still OOM)");
                Ok(())
            }
            _ => Err(CoreStubError::from(crate::core::NexoraError::VramShort {
                required_mb: est_mb,
                available_mb: free.unwrap_or(0),
            })),
        },
    }
}

/// SHA-256 hex of a completed download, streamed in 1 MiB chunks
/// (multi-GB weights are never buffered whole).
///
/// Deviation note: `src/storage` exposes `content_hash_hex(&[u8])` (read-only
/// use per scope), but it requires the full bytes in memory — streaming here
/// with the same SHA-256 algorithm keeps peak RSS flat during install.
async fn sha256_file(path: &std::path::Path) -> Result<String, CoreStubError> {
    let mut file = tokio::fs::File::open(path).await.map_err(|e| {
        CoreStubError::coded(
            "E_HASH_FAILED",
            format!("cannot open {} for hashing: {e}", path.display()),
            "Re-run install; if the file is missing, the download did not finish — check Downloads and disk space.",
        )
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).await.map_err(|e| {
            CoreStubError::coded(
                "E_HASH_FAILED",
                format!("cannot hash {}: {e}", path.display()),
                "The file may be truncated — re-run install to re-fetch it, then check disk health if it repeats.",
            )
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

async fn apply_schema(pool: &SqlitePool) -> Result<(), CoreStubError> {
    apply_migrations(pool).await
}
