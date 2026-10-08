//! Job scheduler: queue + VRAM-gated concurrency (docs/07 §7.5, docs/10).

pub mod queue;

pub use queue::{Job, JobState, Scheduler};
