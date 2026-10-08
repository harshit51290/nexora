//! Batch jobs (`docs/10-API-CLI.md` §10.4).
//!
//! Queue states `Waiting/Running/Completed/Failed/Cancelled`; batch shape
//! `100 prompts -> runner -> model -> 100 images` (datasets/thumbnails).
//!
//! NOTE — complementary queue: `src/scheduler` owns VRAM-gated admission
//! order plus runtime status driving (`SchedulerDriver`: dispatch/complete/
//! fail/cancel with `generations` persistence); this file owns batch
//! submission (`BatchSpec` validate/expand) plus the `JobQueue` facade for
//! the API/CLI/tests, backed by the `jobs` table (`migrations/002_jobs.sql`).
//! Bridge a job over with [`Job::to_scheduler_job`].
//!
//! Persistence contract (same pool pattern as [`crate::model::ModelManager`]
//! and [`crate::scheduler::SchedulerDriver`]: the pool is passed in, never
//! opened here):
//!
//! * the `jobs` table is the truth; the in-memory `Vec<Job>` is a fast-path
//!   cache. Every `*_persisted` enqueue/transition writes through to the DB
//!   first, then mirrors into the cache.
//! * `uar jobs` (and any future API view) reads persisted state via
//!   [`JobQueue::list_persisted`] / [`JobQueue::get_persisted`] — no live
//!   queue handle required.
//! * the table is minimal on purpose (`id, kind, status, model, prompt,
//!   created_at`): `seq` is derived from the `job-NNN` id suffix on reload,
//!   `output_paths` are transient (terminal artifacts live in `generations`),
//!   and VRAM estimates are re-derived at dispatch time.

use crate::core::{NexoraError, Result};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

/// Canonical `jobs` schema (single source stays `migrations/002_jobs.sql`).
/// Replayed with `IF NOT EXISTS` statements only — same statement-split
/// approach as the API bootstrap over `001_init.sql`.
pub const JOBS_SCHEMA_SQL: &str = include_str!("../migrations/002_jobs.sql");

/// Apply the `jobs` schema to `pool`. Idempotent; call once at bootstrap
/// (alongside `001_init.sql`) and in tests.
pub async fn apply_jobs_schema(pool: &SqlitePool) -> Result<()> {
    for stmt in JOBS_SCHEMA_SQL.split(';') {
        let stmt = stmt.trim();
        if stmt.is_empty() {
            continue;
        }
        sqlx::query(stmt).execute(pool).await?;
    }
    Ok(())
}

/// Job families from docs/10 §10.4 (`Job 001 Image / 002 TTS / 003 LLM`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    Image,
    Tts,
    Llm,
}

impl JobKind {
    /// DB spelling (matches the serde JSON spelling).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Tts => "tts",
            Self::Llm => "llm",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "image" => Self::Image,
            "tts" => Self::Tts,
            _ => Self::Llm,
        }
    }
}

/// Queue states from docs/10 §10.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Waiting,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    /// DB spelling: capitalized `Waiting/...` to match
    /// [`crate::scheduler::JobState::as_str`], the `generations.params`
    /// status payloads, and docs/10 §10.4.
    pub fn db_str(self) -> &'static str {
        match self {
            Self::Waiting => "Waiting",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }

    /// Accepts the capitalized DB spelling (and the lowercase serde
    /// spelling, defensively); unknown values fall back to `Waiting`.
    fn from_db_str(s: &str) -> Self {
        match s {
            "Waiting" | "waiting" => Self::Waiting,
            "Running" | "running" => Self::Running,
            "Completed" | "completed" => Self::Completed,
            "Failed" | "failed" => Self::Failed,
            "Cancelled" | "cancelled" => Self::Cancelled,
            _ => Self::Waiting,
        }
    }

    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// `job-001`-style sequence id (human-friendly; not the DB row id).
    pub id: String,
    pub seq: u64,
    pub kind: JobKind,
    pub status: JobStatus,
    pub model: Option<String>,
    pub prompt: String,
    pub created_at: String,
    pub output_paths: Vec<String>,
}

/// `N prompts x M outputs` batch (e.g. 100 prompts -> 100 images).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSpec {
    pub name: String,
    pub model: String,
    pub kind: JobKind,
    pub prompts: Vec<String>,
    pub outputs_per_prompt: u32,
}

impl BatchSpec {
    /// Local validation — real logic, runs before any scheduler exists.
    pub fn validate(&self) -> Result<(), String> {
        if self.prompts.is_empty() {
            return Err("Batch has no prompts (need >= 1 line).".to_string());
        }
        if self.outputs_per_prompt == 0 {
            return Err("outputs_per_prompt must be >= 1.".to_string());
        }
        if self.model.trim().is_empty() {
            return Err("Batch model is empty.".to_string());
        }
        Ok(())
    }

    /// Total outputs this batch will produce.
    pub fn total_outputs(&self) -> usize {
        self.prompts.len() * self.outputs_per_prompt as usize
    }

    /// Expand into one `Waiting` job per (prompt, output) pair. Pure
    /// constructor with relative `job-NNN` ids — persisted entry points
    /// ([`JobQueue::submit_batch_persisted`],
    /// `crate::scheduler::SchedulerDriver::enqueue_batch`) renumber past the
    /// stored max so batches never collide, and admission stays solely with
    /// the scheduler VRAM gate (no second gate here).
    pub fn expand(&self) -> Vec<Job> {
        let mut jobs = Vec::with_capacity(self.total_outputs());
        let mut seq = 0u64;
        for prompt in &self.prompts {
            for _ in 0..self.outputs_per_prompt {
                seq += 1;
                jobs.push(Job {
                    id: format!("job-{seq:03}"),
                    seq,
                    kind: self.kind,
                    status: JobStatus::Waiting,
                    model: Some(self.model.clone()),
                    prompt: prompt.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    output_paths: Vec::new(),
                });
            }
        }
        jobs
    }
}

impl Job {
    /// Numeric suffix of `job-NNN` ids (0 for foreign ids). The `jobs` table
    /// stores no `seq` column (minimal schema); it is re-derived on reload so
    /// the queue counter can resume past the stored max.
    fn seq_from_id(id: &str) -> u64 {
        id.strip_prefix("job-")
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(0)
    }

    /// Convert to the scheduler's runtime job. The scheduler only needs an
    /// id, a model address, and a VRAM estimate — `est_vram_mb` defaults to
    /// `0` when unknown (admits freely under a budget; pass a real estimate
    /// from the analyzer/HW profile whenever one exists).
    pub fn to_scheduler_job(&self, est_vram_mb: Option<u64>) -> crate::scheduler::Job {
        crate::scheduler::Job {
            id: self.id.clone(),
            model_id: self.model.clone().unwrap_or_default(),
            est_vram_mb: est_vram_mb.unwrap_or(0),
            state: match self.status {
                JobStatus::Waiting => crate::scheduler::JobState::Waiting,
                JobStatus::Running => crate::scheduler::JobState::Running,
                JobStatus::Completed => crate::scheduler::JobState::Completed,
                JobStatus::Failed => crate::scheduler::JobState::Failed,
                JobStatus::Cancelled => crate::scheduler::JobState::Cancelled,
            },
        }
    }
}

/// Canonical `jobs` writes, shared by [`JobQueue`] and
/// `crate::scheduler::SchedulerDriver` so every enqueue/transition funnels
/// through one write path (DB truth, memory cache).
///
/// Insert one full-fidelity queue row; `DO NOTHING` on id collision so a
/// scheduler-side bare upsert of an already-stored id never clobbers it.
pub(crate) async fn insert_job_row(pool: &SqlitePool, job: &Job) -> Result<()> {
    sqlx::query(
        "INSERT INTO jobs(id, kind, status, model, prompt, created_at) VALUES(?,?,?,?,?,?)
         ON CONFLICT(id) DO NOTHING",
    )
    .bind(&job.id)
    .bind(job.kind.as_str())
    .bind(job.status.db_str())
    .bind(&job.model)
    .bind(&job.prompt)
    .bind(&job.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

/// Minimal upsert for scheduler-only jobs (no kind/prompt known): creates a
/// `Waiting` row when absent, never touches an existing full-fidelity row.
pub(crate) async fn insert_bare_row(
    pool: &SqlitePool,
    id: &str,
    model: &str,
    created_at: &str,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO jobs(id, kind, status, model, prompt, created_at) VALUES(?, 'llm', 'Waiting', ?, '', ?)
         ON CONFLICT(id) DO NOTHING",
    )
    .bind(id)
    .bind(model)
    .bind(created_at)
    .execute(pool)
    .await?;
    Ok(())
}

/// Persist a status transition. Returns `false` for unknown ids.
pub(crate) async fn update_job_status(
    pool: &SqlitePool,
    id: &str,
    status: JobStatus,
) -> Result<bool> {
    let r = sqlx::query("UPDATE jobs SET status = ? WHERE id = ?")
        .bind(status.db_str())
        .bind(id)
        .execute(pool)
        .await?;
    Ok(r.rows_affected() > 0)
}

/// Highest `job-NNN` suffix persisted so far (0 when none). Persisted batch
/// entry renumbers relative [`BatchSpec::expand`] ids past this so batches
/// (and restarts) never collide.
pub(crate) async fn max_job_seq(pool: &SqlitePool) -> Result<u64> {
    use sqlx::Row;
    let rows = sqlx::query("SELECT id FROM jobs").fetch_all(pool).await?;
    Ok(rows
        .into_iter()
        .filter_map(|r| {
            let id: String = r.get("id");
            let n = Job::seq_from_id(&id);
            if n > 0 {
                Some(n)
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0))
}

fn row_to_job(r: sqlx::sqlite::SqliteRow) -> Job {
    use sqlx::Row;
    let id: String = r.get("id");
    let kind: String = r.get("kind");
    let status: String = r.get("status");
    let model: Option<String> = r.get("model");
    let prompt: String = r.get("prompt");
    let created_at: String = r.get("created_at");
    Job {
        seq: Job::seq_from_id(&id),
        id,
        kind: JobKind::from_str(&kind),
        status: JobStatus::from_db_str(&status),
        model,
        prompt,
        created_at,
        output_paths: Vec::new(),
    }
}

async fn fetch_all_jobs(pool: &SqlitePool) -> Result<Vec<Job>> {
    let rows = sqlx::query(
        "SELECT id, kind, status, model, prompt, created_at FROM jobs ORDER BY created_at, id",
    )
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(row_to_job).collect())
}

async fn fetch_one_job(pool: &SqlitePool, id: &str) -> Result<Option<Job>> {
    let row = sqlx::query(
        "SELECT id, kind, status, model, prompt, created_at FROM jobs WHERE id = ?",
    )
    .bind(id)
    .fetch_optional(pool)
    .await?;
    Ok(row.map(row_to_job))
}

/// Batch-submission facade. Owns [`BatchSpec`] validate/expand plus
/// the prompt/model rows the API/CLI submit; the scheduler
/// (`src/scheduler`) owns VRAM-gated admission order and runtime status
/// driving — hand jobs over via [`Job::to_scheduler_job`].
///
/// Two modes: plain [`JobQueue::new`] is a cache-only stub (sync methods);
/// [`JobQueue::with_pool`] is the persisted queue — the `*_persisted`
/// methods write every enqueue/transition through to the `jobs` table (truth)
/// and mirror into memory (cache).
#[derive(Debug, Default)]
pub struct JobQueue {
    jobs: Vec<Job>,
    next_seq: u64,
    pool: Option<SqlitePool>,
}

impl JobQueue {
    /// Cache-only queue (no pool — sync methods touch memory only).
    pub fn new() -> Self {
        Self::default()
    }

    /// Persisted queue: the pool-backed `jobs` table is the truth, memory is
    /// the cache. Same pool pattern as [`crate::model::ModelManager`].
    pub fn with_pool(pool: SqlitePool) -> Self {
        Self {
            jobs: Vec::new(),
            next_seq: 0,
            pool: Some(pool),
        }
    }

    fn pool(&self) -> Result<&SqlitePool> {
        self.pool.as_ref().ok_or_else(|| {
            NexoraError::Other(anyhow::anyhow!(
                "JobQueue has no pool (cache-only): use JobQueue::with_pool for persisted state"
            ))
        })
    }

    /// Submit one prompt; returns the queued job (always `Waiting` here —
    /// runtime status is driven after handoff via [`Job::to_scheduler_job`]
    /// into the [`crate::scheduler::SchedulerDriver`]).
    ///
    /// Cache-only: touches memory, never the DB. Use
    /// [`JobQueue::submit_persisted`] for the write-through runtime path.
    pub fn submit(
        &mut self,
        kind: JobKind,
        model: Option<String>,
        prompt: String,
    ) -> &Job {
        self.next_seq += 1;
        let seq = self.next_seq;
        self.jobs.push(Job {
            id: format!("job-{seq:03}"),
            seq,
            kind,
            status: JobStatus::Waiting,
            model,
            prompt,
            created_at: chrono::Utc::now().to_rfc3339(),
            output_paths: Vec::new(),
        });
        &self.jobs[self.jobs.len() - 1]
    }

    /// Submit a whole batch; returns the queued job ids. Ids always come
    /// from the queue counter (never from [`BatchSpec::expand`], whose
    /// numbering is relative) so mixed `submit`/`submit_batch` use never
    /// collides.
    ///
    /// Cache-only: touches memory, never the DB. Use
    /// [`JobQueue::submit_batch_persisted`] for the write-through runtime path.
    pub fn submit_batch(&mut self, spec: &BatchSpec) -> Result<Vec<String>, String> {
        spec.validate()?;
        let mut ids = Vec::with_capacity(spec.total_outputs());
        for prompt in &spec.prompts {
            for _ in 0..spec.outputs_per_prompt {
                let id = self
                    .submit(spec.kind, Some(spec.model.clone()), prompt.clone())
                    .id
                    .clone();
                ids.push(id);
            }
        }
        Ok(ids)
    }

    pub fn list(&self) -> &[Job] {
        &self.jobs
    }

    pub fn get(&self, id: &str) -> Option<&Job> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Test/local hook for status transitions. Returns false for unknown ids.
    /// At runtime the [`crate::scheduler::SchedulerDriver`] drives these
    /// transitions and persists terminal rows to `generations`.
    ///
    /// Cache-only: touches memory, never the DB. Use
    /// [`JobQueue::set_status_persisted`] for the write-through runtime path.
    pub fn set_status(&mut self, id: &str, status: JobStatus) -> bool {
        match self.jobs.iter_mut().find(|j| j.id == id) {
            Some(job) => {
                job.status = status;
                true
            }
            None => false,
        }
    }

    /// Cache-only cancel. Use [`JobQueue::cancel_persisted`] for the
    /// write-through runtime path.
    pub fn cancel(&mut self, id: &str) -> bool {
        self.set_status(id, JobStatus::Cancelled)
    }

    // -- persisted (DB truth, memory cache) --------------------------------

    /// Persisted submit: insert the `Waiting` row (DB truth), then mirror
    /// into the cache. Ids still come from the queue counter.
    pub async fn submit_persisted(
        &mut self,
        kind: JobKind,
        model: Option<String>,
        prompt: String,
    ) -> Result<Job> {
        self.next_seq += 1;
        let seq = self.next_seq;
        let job = Job {
            id: format!("job-{seq:03}"),
            seq,
            kind,
            status: JobStatus::Waiting,
            model,
            prompt,
            created_at: chrono::Utc::now().to_rfc3339(),
            output_paths: Vec::new(),
        };
        let pool = self.pool()?.clone();
        insert_job_row(&pool, &job).await?;
        self.jobs.push(job.clone());
        Ok(job)
    }

    /// Persisted batch: validate (no VRAM gate here — admission stays solely
    /// with the scheduler gate), expand, and enqueue one persisted `Waiting`
    /// row per (prompt, output) pair. Returns the queued ids.
    pub async fn submit_batch_persisted(&mut self, spec: &BatchSpec) -> Result<Vec<String>> {
        spec.validate()
            .map_err(|e| NexoraError::Other(anyhow::anyhow!("{e}")))?;
        let mut ids = Vec::with_capacity(spec.total_outputs());
        for prompt in &spec.prompts {
            for _ in 0..spec.outputs_per_prompt {
                let job = self
                    .submit_persisted(spec.kind, Some(spec.model.clone()), prompt.clone())
                    .await?;
                ids.push(job.id);
            }
        }
        Ok(ids)
    }

    /// Persisted transition: write the status through, then mirror to cache.
    /// Returns `false` for unknown ids. Terminal rows are sticky: a row
    /// already `Completed`/`Failed`/`Cancelled` keeps its terminal state
    /// (the cached copy is re-synced to the truth, `false` returned) so the
    /// queue history never moves backwards.
    pub async fn set_status_persisted(&mut self, id: &str, status: JobStatus) -> Result<bool> {
        let pool = self.pool()?.clone();
        if let Some(cur) = fetch_one_job(&pool, id).await? {
            if cur.status.is_terminal() && cur.status != status {
                self.mirror(cur);
                return Ok(false);
            }
        }
        if !update_job_status(&pool, id, status).await? {
            return Ok(false);
        }
        if let Some(slot) = self.jobs.iter_mut().find(|j| j.id == id) {
            slot.status = status;
        }
        Ok(true)
    }

    /// Persisted cancel: `Waiting`/`Running` -> `Cancelled`. Terminal rows
    /// stay untouched (`false`), unknown ids return `false`.
    pub async fn cancel_persisted(&mut self, id: &str) -> Result<bool> {
        let pool = self.pool()?.clone();
        match fetch_one_job(&pool, id).await? {
            Some(cur)
                if matches!(cur.status, JobStatus::Waiting | JobStatus::Running) =>
            {
                self.set_status_persisted(id, JobStatus::Cancelled).await
            }
            _ => Ok(false),
        }
    }

    /// Reload the cache from the DB truth (order/state as persisted; `seq`
    /// re-derived, counter resumed past the max stored so later submits never
    /// collide — including across restarts). After this, [`JobQueue::list`]
    /// shows persisted — not in-memory — state.
    pub async fn reload(&mut self) -> Result<()> {
        let pool = self.pool()?.clone();
        let rows = fetch_all_jobs(&pool).await?;
        let stored_max = rows.iter().map(|j| j.seq).max().unwrap_or(0);
        self.next_seq = self.next_seq.max(stored_max);
        self.jobs = rows;
        Ok(())
    }

    /// Read persisted queue state without a queue handle — the `uar jobs`
    /// view path (DB truth, no cache involved).
    pub async fn list_persisted(pool: &SqlitePool) -> Result<Vec<Job>> {
        fetch_all_jobs(pool).await
    }

    /// Read one persisted job without a queue handle. `None` for unknown ids.
    pub async fn get_persisted(pool: &SqlitePool, id: &str) -> Result<Option<Job>> {
        fetch_one_job(pool, id).await
    }

    fn mirror(&mut self, job: Job) {
        match self.jobs.iter_mut().find(|j| j.id == job.id) {
            Some(slot) => *slot = job,
            None => {
                self.next_seq = self.next_seq.max(job.seq);
                self.jobs.push(job);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn mem_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        apply_jobs_schema(&pool).await.unwrap();
        pool
    }

    #[tokio::test]
    async fn enqueue_persists_and_reload_shows_rows() {
        let pool = mem_pool().await;
        let mut q = JobQueue::with_pool(pool.clone());
        let a = q
            .submit_persisted(JobKind::Image, Some("m".into()), "a prompt".into())
            .await
            .unwrap();
        let b = q
            .submit_persisted(JobKind::Tts, None, "b prompt".into())
            .await
            .unwrap();
        assert_eq!(q.list().len(), 2);

        // Fresh handle over the same DB: reload shows persisted state.
        let mut q2 = JobQueue::with_pool(pool.clone());
        assert!(q2.list().is_empty());
        q2.reload().await.unwrap();
        assert_eq!(q2.list().len(), 2);
        assert_eq!(q2.get(&a.id).unwrap().prompt, "a prompt");
        assert_eq!(q2.get(&b.id).unwrap().status, JobStatus::Waiting);
        assert_eq!(q2.get(&a.id).unwrap().kind, JobKind::Image);

        // A persisted transition survives another reload.
        assert!(q2.set_status_persisted(&a.id, JobStatus::Running).await.unwrap());
        let mut q3 = JobQueue::with_pool(pool.clone());
        q3.reload().await.unwrap();
        assert_eq!(q3.get(&a.id).unwrap().status, JobStatus::Running);

        // Static view path for `uar jobs`: no handle needed.
        let rows = JobQueue::list_persisted(&pool).await.unwrap();
        assert_eq!(rows.len(), 2);

        // The id counter resumes past the stored max — no collision.
        let c = q3
            .submit_persisted(JobKind::Llm, None, "c".into())
            .await
            .unwrap();
        assert_ne!(c.id, a.id);
        assert_ne!(c.id, b.id);
    }

    #[tokio::test]
    async fn cancel_paths() {
        let pool = mem_pool().await;
        let mut q = JobQueue::with_pool(pool.clone());
        let w = q
            .submit_persisted(JobKind::Image, None, "w".into())
            .await
            .unwrap();
        assert!(q.cancel_persisted(&w.id).await.unwrap());
        assert_eq!(
            JobQueue::get_persisted(&pool, &w.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            JobStatus::Cancelled
        );
        // Already terminal -> false; row keeps its terminal state.
        assert!(!q.cancel_persisted(&w.id).await.unwrap());
        // Unknown id -> false.
        assert!(!q.cancel_persisted("job-999").await.unwrap());
        assert!(JobQueue::get_persisted(&pool, "job-999").await.unwrap().is_none());
        // Terminal rows are sticky under set_status too.
        assert!(!q.set_status_persisted(&w.id, JobStatus::Waiting).await.unwrap());
        assert_eq!(
            JobQueue::get_persisted(&pool, &w.id)
                .await
                .unwrap()
                .unwrap()
                .status,
            JobStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn batch_expansion_enqueues_persisted_rows() {
        let pool = mem_pool().await;
        let mut q = JobQueue::with_pool(pool.clone());
        let spec = BatchSpec {
            name: "t".into(),
            model: "m".into(),
            kind: JobKind::Image,
            prompts: vec!["p1".into(), "p2".into()],
            outputs_per_prompt: 2,
        };
        let ids = q.submit_batch_persisted(&spec).await.unwrap();
        assert_eq!(ids.len(), 4);
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);

        let mut q2 = JobQueue::with_pool(pool.clone());
        q2.reload().await.unwrap();
        assert_eq!(q2.list().len(), 4);
        assert!(q2.list().iter().all(|j| j.status == JobStatus::Waiting));
        assert!(q2.list().iter().all(|j| j.model.as_deref() == Some("m")));

        // Invalid specs are rejected before any row is written.
        let bad = BatchSpec {
            name: "bad".into(),
            model: "m".into(),
            kind: JobKind::Image,
            prompts: vec![],
            outputs_per_prompt: 1,
        };
        assert!(q.submit_batch_persisted(&bad).await.is_err());
        assert_eq!(JobQueue::list_persisted(&pool).await.unwrap().len(), 4);
    }
}
