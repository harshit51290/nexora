//! Batch jobs (`docs/10-API-CLI.md` §10.4).
//!
//! Queue states `Waiting/Running/Completed/Failed/Cancelled`; batch shape
//! `100 prompts -> runner -> model -> 100 images` (datasets/thumbnails).
//!
//! NOTE — complementary queue: `src/scheduler` owns VRAM-gated admission
//! order; this file owns batch submission (`BatchSpec` validate/expand) plus a
//! minimal in-memory `JobQueue` for the API/CLI/tests until persistence lands.
//! TODO-WIRE: scheduler drives `set_status` at runtime; persist rows via
//! `generations`.

use serde::{Deserialize, Serialize};

/// Job families from docs/10 §10.4 (`Job 001 Image / 002 TTS / 003 LLM`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    Image,
    Tts,
    Llm,
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

    /// Expand into one `Waiting` job per (prompt, output) pair.
    /// TODO-CORE-WIRE: the scheduler owns expansion + VRAM gating at runtime.
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

/// In-memory stub queue. See module NOTE: delete when `src/scheduler` lands.
#[derive(Debug, Default)]
pub struct JobQueue {
    jobs: Vec<Job>,
    next_seq: u64,
}

impl JobQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit one prompt; returns the queued job (always `Waiting` here —
    /// no executor runs until the scheduler lands).
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

    /// Test/scheduler-driver hook for status transitions.
    /// Returns false for unknown ids. TODO-CORE-WIRE: scheduler drives this.
    pub fn set_status(&mut self, id: &str, status: JobStatus) -> bool {
        match self.jobs.iter_mut().find(|j| j.id == id) {
            Some(job) => {
                job.status = status;
                true
            }
            None => false,
        }
    }

    pub fn cancel(&mut self, id: &str) -> bool {
        self.set_status(id, JobStatus::Cancelled)
    }
}
