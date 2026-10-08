//! Jobs: `Waiting -> Running -> Completed | Failed | Cancelled`.
//! A job starts only if the VRAM-gated concurrency check passes:
//! concurrent models (LLM+TTS+Image) are allowed only when the combined
//! estimate fits (docs/07 §7.5). Jobs never execute synchronously here —
//! workers pick up `Running` jobs and drive child-process runtimes.

use crate::core::{NexoraError, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobState {
    Waiting,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Waiting => "Waiting",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub model_id: String,
    pub est_vram_mb: u64,
    pub state: JobState,
}

/// In-memory queue with a VRAM budget. This type owns admission order and is
/// the single admission authority (the VRAM gate in [`Scheduler::start_next`]);
/// persistence lives one layer up — [`super::driver::SchedulerDriver`] writes
/// every enqueue/transition through to the `jobs` table, so this in-memory
/// state is a fast-path cache and the DB is the truth.
#[derive(Debug, Default)]
pub struct Scheduler {
    waiting: VecDeque<Job>,
    running: Vec<Job>,
    pub max_concurrent: usize,
    pub vram_budget_mb: Option<u64>,
}

impl Scheduler {
    pub fn new(max_concurrent: usize, vram_budget_mb: Option<u64>) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            max_concurrent: max_concurrent.max(1),
            vram_budget_mb,
        }
    }

    pub fn enqueue(&mut self, job: Job) {
        self.waiting.push_back(job);
    }

    fn running_vram(&self) -> u64 {
        self.running.iter().map(|j| j.est_vram_mb).sum()
    }

    /// VRAM-gated admission: next job starts only if a slot is free AND
    /// its estimate fits the remaining budget (`E_VRAM_SHORT` otherwise).
    pub fn start_next(&mut self) -> Result<Option<Job>> {
        if self.running.len() >= self.max_concurrent {
            return Ok(None);
        }
        let next = match self.waiting.front() {
            Some(j) => j.clone(),
            None => return Ok(None),
        };
        if let Some(budget) = self.vram_budget_mb {
            let used = self.running_vram();
            let free = budget.saturating_sub(used);
            if next.est_vram_mb > free {
                return Err(NexoraError::VramShort {
                    required_mb: next.est_vram_mb,
                    available_mb: free,
                });
            }
        }
        let mut job = self.waiting.pop_front().expect("checked above");
        job.state = JobState::Running;
        self.running.push(job.clone());
        Ok(Some(job))
    }

    pub fn complete(&mut self, id: &str) -> bool {
        self.finish(id, JobState::Completed)
    }
    pub fn fail(&mut self, id: &str) -> bool {
        self.finish(id, JobState::Failed)
    }
    pub fn cancel(&mut self, id: &str) -> bool {
        if self.cancel_waiting(id) {
            return true;
        }
        self.cancel_running(id)
    }

    /// Drop a `Waiting` job silently (never ran — no terminal row to persist).
    pub fn cancel_waiting(&mut self, id: &str) -> bool {
        if let Some(pos) = self.waiting.iter().position(|j| j.id == id) {
            self.waiting.remove(pos);
            return true;
        }
        false
    }

    /// Move a `Running` job to `Cancelled` (caller persists the terminal row).
    pub fn cancel_running(&mut self, id: &str) -> bool {
        self.finish(id, JobState::Cancelled)
    }

    fn finish(&mut self, id: &str, _state: JobState) -> bool {
        if let Some(pos) = self.running.iter().position(|j| j.id == id) {
            self.running.remove(pos);
            // Terminal transition = removal from `running`; the service layer
            // persists the final state to `generations`. (`Job` is `Clone`;
            // the queued copy already carried `Running`.)
            return true;
        }
        false
    }

    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// Drop all queued state. Used by persisted reload: the DB truth is
    /// re-imported right after via [`Scheduler::import`].
    pub fn clear(&mut self) {
        self.waiting.clear();
        self.running.clear();
    }

    /// Re-import one row from the persisted truth into its matching lane.
    /// Terminal states are ignored (history only — never re-queued).
    pub fn import(&mut self, job: Job) {
        match job.state {
            JobState::Waiting => self.waiting.push_back(job),
            JobState::Running => self.running.push(job),
            _ => {}
        }
    }
}
