//! Runtime status driver over [`Scheduler`](super::queue::Scheduler).
//!
//! The in-memory `Scheduler` owns VRAM-gated admission order
//! (`Waiting -> Running`) — the single admission authority, no second gate;
//! this driver owns runtime status driving
//! (`Running -> Completed | Failed | Cancelled`) and persists every terminal
//! transition as a row in `generations` (docs/10 §10.4, docs/04 §4.1).
//!
//! Queue persistence: every enqueue/transition additionally writes through
//! to the `jobs` table (`migrations/002_jobs.sql`) — the in-memory queue is
//! a fast-path cache, the DB is the truth. Fresh drivers rehydrate via
//! [`SchedulerDriver::reload`]; [`SchedulerDriver::list_persisted`] is the
//! `uar jobs` view path (no live handle required). Batch entry goes through
//! [`SchedulerDriver::enqueue_batch`] (validate + expand + persisted enqueue;
//! admission still happens only at the VRAM gate).
//!
//! The [`SqlitePool`] is passed in — same pool pattern as
//! [`crate::model::ModelManager`]; this module never opens its own connection.
//! Jobs never execute synchronously here beyond the injected executor: pass a
//! closure that dispatches to a worker / child-process runtime
//! (orchestration-only, AGENTS.md rule 1). Production workers pick up
//! `Running` jobs and drive the runtime adapters; the driver only moves state
//! and records rows.

use super::queue::{Job, JobState, Scheduler};
use crate::core::{NexoraError, Result};
use crate::jobs::{BatchSpec, JobStatus};
use sqlx::SqlitePool;

/// Outcome of one executed job, recorded into `generations`.
#[derive(Debug, Clone, Default)]
pub struct JobOutcome {
    /// File produced by the run, if any (`generations.output_path`).
    pub output_path: Option<String>,
    /// Prompt that produced it (`generations.prompt`).
    pub prompt: Option<String>,
    /// Params sidecar JSON (`generations.params`).
    pub params_json: Option<String>,
    /// Adapter that ran it, e.g. `"diffusers"` (`generations.runtime`).
    pub runtime: Option<String>,
}

/// Why an executor failed: machine-readable code + detail.
/// Persisted into `generations.params` and `logs` (AGENTS.md rule 7).
#[derive(Debug, Clone)]
pub struct ExecFailure {
    pub code: String,
    pub message: String,
}

impl ExecFailure {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Per-job result of [`SchedulerDriver::drain`].
#[derive(Debug, Clone)]
pub enum DrainResult {
    Completed { id: String },
    Failed { id: String, failure: ExecFailure },
}

/// Moves jobs through their lifecycle on top of the VRAM-gated [`Scheduler`].
pub struct SchedulerDriver {
    sched: Scheduler,
    pool: SqlitePool,
}

impl SchedulerDriver {
    pub fn new(max_concurrent: usize, vram_budget_mb: Option<u64>, pool: SqlitePool) -> Self {
        Self {
            sched: Scheduler::new(max_concurrent, vram_budget_mb),
            pool,
        }
    }

    /// Queue a job as `Waiting`: in-memory cache plus a `jobs` write-through
    /// (minimal bare row when the id is new; a full-fidelity `JobQueue` row
    /// always wins on conflict).
    pub async fn enqueue(&mut self, job: Job) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        crate::jobs::insert_bare_row(&self.pool, &job.id, &job.model_id, &now).await?;
        self.sched.enqueue(job);
        Ok(())
    }

    /// Batch entry: validate (only — admission stays solely with the VRAM
    /// gate in [`Scheduler::start_next`], never here), expand, and enqueue
    /// one persisted `Waiting` row per (prompt, output) pair. Relative
    /// [`BatchSpec::expand`] ids are renumbered past the stored max so
    /// batches (and restarts) never collide. Full kind/prompt rows are
    /// inserted first, so the later bare upserts are no-ops.
    pub async fn enqueue_batch(
        &mut self,
        spec: &BatchSpec,
        est_vram_mb: Option<u64>,
    ) -> Result<Vec<String>> {
        spec.validate()
            .map_err(|e| NexoraError::Other(anyhow::anyhow!("{e}")))?;
        let mut seq = crate::jobs::max_job_seq(&self.pool).await?;
        let mut ids = Vec::with_capacity(spec.total_outputs());
        for mut record in spec.expand() {
            seq += 1;
            record.id = format!("job-{seq:03}");
            record.seq = seq;
            record.created_at = chrono::Utc::now().to_rfc3339();
            crate::jobs::insert_job_row(&self.pool, &record).await?;
            self.sched.enqueue(record.to_scheduler_job(est_vram_mb));
            ids.push(format!("job-{seq:03}"));
        }
        Ok(ids)
    }

    pub fn waiting(&self) -> usize {
        self.sched.waiting()
    }

    pub fn running(&self) -> usize {
        self.sched.running()
    }

    /// Admit the next job (`Waiting -> Running`) through the existing VRAM
    /// gate. `Ok(None)` = nothing admissible right now; `Err(VramShort)` =
    /// head-of-line job does not fit the remaining budget. Persists the
    /// `Running` transition to the `jobs` truth.
    pub async fn dispatch_next(&mut self) -> Result<Option<Job>> {
        let next = self.sched.start_next()?;
        if let Some(ref job) = next {
            crate::jobs::update_job_status(&self.pool, &job.id, JobStatus::Running).await?;
        }
        Ok(next)
    }

    /// Rebuild the in-memory cache from the DB truth: `Waiting` rows go back
    /// to waiting, `Running` rows back to running (estimates are transient —
    /// reloaded rows admit with `est_vram_mb = 0`, re-derive estimates at
    /// dispatch). Terminal rows stay history-only. Call once after `new()`
    /// at bootstrap to pick up rows persisted by a previous process.
    pub async fn reload(&mut self) -> Result<()> {
        self.sched.clear();
        for record in crate::jobs::JobQueue::list_persisted(&self.pool).await? {
            match record.status {
                JobStatus::Waiting | JobStatus::Running => {
                    self.sched.import(record.to_scheduler_job(None));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Persisted queue view — what `uar jobs` displays (DB truth, no cache).
    pub async fn list_persisted(&self) -> Result<Vec<crate::jobs::Job>> {
        crate::jobs::JobQueue::list_persisted(&self.pool).await
    }

    /// One persisted job by id. `None` for unknown ids.
    pub async fn get_persisted(&self, id: &str) -> Result<Option<crate::jobs::Job>> {
        crate::jobs::JobQueue::get_persisted(&self.pool, id).await
    }

    /// `Running -> Completed`, then persist the terminal row to `generations`.
    /// Returns `false` when `job.id` is not currently running. The `jobs`
    /// queue row flips first (queue truth), the `generations` artifact row
    /// second.
    pub async fn complete(&mut self, job: &Job, outcome: JobOutcome) -> Result<bool> {
        if !self.sched.complete(&job.id) {
            return Ok(false);
        }
        crate::jobs::update_job_status(&self.pool, &job.id, JobStatus::Completed).await?;
        self.record_terminal(job, None, &outcome, "Completed").await?;
        Ok(true)
    }

    /// `Running -> Failed`, then persist the terminal row (error code inside
    /// `generations.params`) plus a `logs` row. Returns `false` when `job.id`
    /// is not currently running. The `jobs` queue row flips first (queue
    /// truth), the `generations` artifact row second.
    pub async fn fail(&mut self, job: &Job, failure: &ExecFailure) -> Result<bool> {
        if !self.sched.fail(&job.id) {
            return Ok(false);
        }
        crate::jobs::update_job_status(&self.pool, &job.id, JobStatus::Failed).await?;
        let outcome = JobOutcome::default();
        self.record_terminal(job, Some(failure), &outcome, "Failed").await?;
        Ok(true)
    }

    /// Cancel by id. A `Waiting` job leaves the queue and its persisted row
    /// flips to `Cancelled` (never ran — no `generations` row). A `Running`
    /// job transitions to `Cancelled` and gets a terminal row. Returns
    /// `false` for unknown ids.
    pub async fn cancel(&mut self, id: &str, model_id: &str) -> Result<bool> {
        if self.sched.cancel_waiting(id) {
            crate::jobs::update_job_status(&self.pool, id, JobStatus::Cancelled).await?;
            return Ok(true);
        }
        if self.sched.cancel_running(id) {
            let job = Job {
                id: id.to_string(),
                model_id: model_id.to_string(),
                est_vram_mb: 0,
                state: JobState::Cancelled,
            };
            crate::jobs::update_job_status(&self.pool, id, JobStatus::Cancelled).await?;
            let outcome = JobOutcome {
                params_json: Some(r#"{"status":"Cancelled"}"#.to_string()),
                ..JobOutcome::default()
            };
            self.record_terminal(&job, None, &outcome, "Cancelled").await?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Drive the queue until nothing is admissible: admit each job through the
    /// VRAM gate, hand it to `exec`, then settle + persist. Stops early when
    /// the head-of-line job no longer fits the remaining budget (`E_VRAM_SHORT`)
    /// — the rest keep waiting for running jobs to finish and free VRAM.
    ///
    /// `exec` is synchronous on purpose: the queue mechanics stay testable
    /// without runtimes; production callers pass a closure that blocks on a
    /// worker / child-process result.
    pub async fn drain<F>(&mut self, mut exec: F) -> Vec<DrainResult>
    where
        F: FnMut(&Job) -> std::result::Result<JobOutcome, ExecFailure>,
    {
        let mut results = Vec::new();
        loop {
            let job = match self.dispatch_next().await {
                Ok(Some(job)) => job,
                Ok(None) => break,
                Err(NexoraError::VramShort { .. }) => break,
                Err(e) => {
                    // `start_next` only fails with `VramShort` today (plus a
                    // possible `E_DB` from the `Running` write-through); any
                    // such error also stops the drain, never skips.
                    tracing::warn!("scheduler drain stopped on gate error: {e}");
                    break;
                }
            };
            match exec(&job) {
                Ok(outcome) => {
                    let id = job.id.clone();
                    match self.complete(&job, outcome).await {
                        Ok(true) => results.push(DrainResult::Completed { id }),
                        Ok(false) => results.push(DrainResult::Failed {
                            id: id.clone(),
                            failure: ExecFailure::new(
                                "E_SCHEDULER_LOST",
                                format!("job {id} left Running before it could complete"),
                            ),
                        }),
                        Err(e) => results.push(DrainResult::Failed {
                            id,
                            failure: ExecFailure::new("E_DB", e.to_string()),
                        }),
                    }
                }
                Err(failure) => {
                    let id = job.id.clone();
                    match self.fail(&job, &failure).await {
                        Ok(_) => results.push(DrainResult::Failed { id, failure }),
                        Err(e) => results.push(DrainResult::Failed {
                            id,
                            failure: ExecFailure::new("E_DB", e.to_string()),
                        }),
                    }
                }
            }
        }
        results
    }

    /// Insert the terminal `generations` row + a `logs` row for one settled job.
    async fn record_terminal(
        &self,
        job: &Job,
        failure: Option<&ExecFailure>,
        outcome: &JobOutcome,
        terminal: &str,
    ) -> Result<()> {
        let (params, level) = match failure {
            Some(f) => (
                serde_json::json!({
                    "status": "Failed",
                    "error_code": f.code,
                    "error": f.message,
                })
                .to_string(),
                "error",
            ),
            None => (
                outcome.params_json.clone().unwrap_or_else(|| {
                    // The in-memory `job` copy still carries `Running` (the
                    // queue drops it on settle), so the persisted status comes
                    // from the caller, never from `job.state`.
                    serde_json::json!({ "status": terminal }).to_string()
                }),
                "info",
            ),
        };
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO generations(id, model_id, prompt, params, seed, output_path, created_at, runtime) VALUES(?,?,?,?,?,?,?,?)",
        )
        .bind(&job.id)
        .bind(&job.model_id)
        .bind(outcome.prompt.clone())
        .bind(params)
        .bind(Option::<i64>::None)
        .bind(outcome.output_path.clone())
        .bind(&now)
        .bind(outcome.runtime.clone())
        .execute(&self.pool)
        .await?;
        let msg = match failure {
            Some(f) => format!("job {} {}: [{}] {}", job.id, job.model_id, f.code, f.message),
            None => format!("job {} {} -> {terminal}", job.id, job.model_id),
        };
        sqlx::query("INSERT INTO logs(ts, scope, level, msg) VALUES(?,?,?,?)")
            .bind(&now)
            .bind("Scheduler")
            .bind(level)
            .bind(msg)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, vram: u64) -> Job {
        Job {
            id: id.to_string(),
            model_id: "m".to_string(),
            est_vram_mb: vram,
            state: super::super::queue::JobState::Waiting,
        }
    }

    async fn mem_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        for schema in [
            include_str!("../../migrations/001_init.sql"),
            include_str!("../../migrations/002_jobs.sql"),
        ] {
            for stmt in schema.split(';') {
                let stmt = stmt.trim();
                if !stmt.is_empty() {
                    sqlx::query(stmt).execute(&pool).await.unwrap();
                }
            }
        }
        // sqlx enforces FKs: jobs/generations reference model 'm'.
        sqlx::query("INSERT INTO models(id, name, repository, status) VALUES('m','m','o/m','READY')")
            .execute(&pool)
            .await
            .unwrap();
        pool
    }

    #[tokio::test]
    async fn drain_completes_and_persists_generations() {
        let pool = mem_pool().await;
        let mut driver = SchedulerDriver::new(2, Some(4096), pool.clone());
        driver.enqueue(job("a", 1000)).await.unwrap();
        driver.enqueue(job("b", 1000)).await.unwrap();
        let results = driver
            .drain(|j| {
                Ok(JobOutcome {
                    prompt: Some(format!("prompt-{}", j.id)),
                    ..JobOutcome::default()
                })
            })
            .await;
        assert_eq!(results.len(), 2);
        assert!(matches!(results[0], DrainResult::Completed { .. }));
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM generations")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 2);
        let done: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM jobs WHERE status = 'Completed'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(done, 2);
        assert_eq!(driver.waiting(), 0);
        assert_eq!(driver.running(), 0);
    }

    #[tokio::test]
    async fn drain_stops_on_vram_gate_and_records_failures() {
        let pool = mem_pool().await;
        let mut driver = SchedulerDriver::new(2, Some(1500), pool.clone());
        driver.enqueue(job("big", 1200)).await.unwrap();
        driver.enqueue(job("too-big", 1200)).await.unwrap();
        driver.enqueue(job("bad", 100)).await.unwrap();
        // "big" fails (budget freed on settle), "too-big" then fits, then "bad".
        let results = driver
            .drain(|j| {
                if j.id == "big" {
                    Err(ExecFailure::new("E_RUNTIME_CRASH", "boom"))
                } else {
                    Ok(JobOutcome::default())
                }
            })
            .await;
        // "big" fails (frees budget), "too-big" runs ok, "bad" runs ok.
        assert_eq!(results.len(), 3);
        assert!(matches!(&results[0], DrainResult::Failed { .. }));
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM generations")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 3);
    }

    #[tokio::test]
    async fn vram_blocked_head_stops_drain() {
        let pool = mem_pool().await;
        let mut driver = SchedulerDriver::new(1, Some(500), pool);
        driver.enqueue(job("huge", 4000)).await.unwrap();
        let results = driver.drain(|_| Ok(JobOutcome::default())).await;
        assert!(results.is_empty());
        assert_eq!(driver.waiting(), 1);
    }

    #[tokio::test]
    async fn enqueue_persists_and_reload_restores_cache() {
        let pool = mem_pool().await;
        let mut driver = SchedulerDriver::new(2, Some(4096), pool.clone());
        driver.enqueue(job("a", 1000)).await.unwrap();
        driver.enqueue(job("b", 1000)).await.unwrap();
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM jobs WHERE status = 'Waiting'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 2);

        // Fresh driver over the same DB reloads persisted (not in-memory) state.
        let mut driver2 = SchedulerDriver::new(2, Some(4096), pool.clone());
        assert_eq!(driver2.waiting(), 0);
        driver2.reload().await.unwrap();
        assert_eq!(driver2.waiting(), 2);

        // Admission still goes through the single VRAM gate and persists Running.
        let admitted = driver2.dispatch_next().await.unwrap().unwrap();
        assert_eq!(admitted.id, "a");
        let st: String = sqlx::query_scalar("SELECT status FROM jobs WHERE id = 'a'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(st, "Running");
    }

    #[tokio::test]
    async fn cancel_paths_persist() {
        let pool = mem_pool().await;
        let mut driver = SchedulerDriver::new(2, Some(4096), pool.clone());
        driver.enqueue(job("w", 100)).await.unwrap();
        driver.enqueue(job("r", 100)).await.unwrap();

        // Waiting cancel: leaves the queue, persisted row flips to Cancelled,
        // no generations row (never ran).
        assert!(driver.cancel("w", "m").await.unwrap());
        assert_eq!(driver.waiting(), 1);
        let st: String = sqlx::query_scalar("SELECT status FROM jobs WHERE id = 'w'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(st, "Cancelled");
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM generations")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 0);

        // Running cancel: terminal row + Cancelled status.
        let admitted = driver.dispatch_next().await.unwrap().unwrap();
        assert_eq!(admitted.id, "r");
        assert!(driver.cancel("r", "m").await.unwrap());
        let st: String = sqlx::query_scalar("SELECT status FROM jobs WHERE id = 'r'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(st, "Cancelled");
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM generations")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 1);

        // Unknown id -> false.
        assert!(!driver.cancel("nope", "m").await.unwrap());
    }

    #[tokio::test]
    async fn batch_expansion_enqueues_persisted_rows_without_second_gate() {
        // Budget far below the estimate: submit must still queue everything
        // Waiting (no gate at submit); the single VRAM gate applies at dispatch.
        let pool = mem_pool().await;
        let mut driver = SchedulerDriver::new(1, Some(1), pool.clone());
        let spec = crate::jobs::BatchSpec {
            name: "t".to_string(),
            model: "m".to_string(),
            kind: crate::jobs::JobKind::Image,
            prompts: vec!["p1".to_string(), "p2".to_string()],
            outputs_per_prompt: 2,
        };
        let ids = driver.enqueue_batch(&spec, Some(100)).await.unwrap();
        assert_eq!(ids.len(), 4);
        assert_eq!(driver.waiting(), 4);
        let rows = driver.list_persisted().await.unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|j| j.prompt == "p1" || j.prompt == "p2"));

        // A second batch never collides with the first.
        let ids2 = driver.enqueue_batch(&spec, Some(100)).await.unwrap();
        assert_eq!(ids2.len(), 4);
        assert!(ids2.iter().all(|id| !ids.contains(id)));
        assert_eq!(driver.list_persisted().await.unwrap().len(), 8);

        // Dispatch is where the gate bites: head needs 100MB, budget is 1MB.
        let results = driver.drain(|_| Ok(JobOutcome::default())).await;
        assert!(results.is_empty());
        assert_eq!(driver.waiting(), 8);
    }
}
