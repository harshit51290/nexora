//! Core primitives: persisted state machines + error catalog.

pub mod error;
pub mod state;

pub use error::{NexoraError, Result};
pub use state::{ModelState, RuntimeState};
