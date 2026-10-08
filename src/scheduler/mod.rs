//! Job scheduler: queue + VRAM-gated concurrency + runtime status driver +
//! node-graph workflow runner (docs/07 §7.5, docs/10 §10.4–10.5).

pub mod driver;
pub mod queue;
pub mod workflow;

pub use driver::{DrainResult, ExecFailure, JobOutcome, SchedulerDriver};
pub use queue::{Job, JobState, Scheduler};
pub use workflow::{
    stub_execute, KNOWN_NODE_TYPES, NodeFailure, WorkflowError, WorkflowNode, WorkflowRun,
    WorkflowTemplate,
};
