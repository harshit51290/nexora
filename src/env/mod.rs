//! Isolated per-runtime Python environments (`docs/09-ENV-SECURITY.md`).
//!
//! Different models need conflicting stacks, so there is never one shared
//! global env. Layout: `environments/<id>/{python,packages,environment.json}`.
//! Compatible models share one env via the resolver
//! ([`EnvManager::resolve_shared_env`], entered through
//! [`EnvManager::resolve_for_runtime`]); incompatible ones get separate envs.
//! Every env pins Python / packages / CUDA / model revision, and the previous
//! pin is kept for one-click rollback.
//!
//! Pin records are `serde`-serialized and [`EnvError`] converts into the core
//! [`crate::core::NexoraError`]. NEED (integrator, out of scope): persist
//! [`EnvRecord`] through the `environments` SQLite table (`docs/04` §4.1).

pub mod manager;

pub use manager::{EnvError, EnvHandle, EnvManager, EnvPin, EnvRecord};
