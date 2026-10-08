//! Model manager: install / remove / load / unload / metadata.
//! Persists every [`crate::core::ModelState`] transition to SQLite.

pub mod manager;

pub use manager::{ModelManager, ModelRecord};
