//! Isolated per-runtime Python environments (`docs/09-ENV-SECURITY.md`).
//!
//! Different models need conflicting stacks, so there is never one shared
//! global env. Layout: `environments/<id>/{python,packages,environment.json}`.
//! Compatible models share one env via the resolver
//! ([`EnvManager::resolve_shared_env`]); incompatible ones get separate envs.
//! Every env pins Python / packages / CUDA / model revision, and the previous
//! pin is kept for one-click rollback.
//!
//! TODO-CORE-ALIGN: unify [`EnvError`] with the core error catalog (see
//! `src/runtime/adapter.rs`), persist [`EnvRecord`] through the
//! `environments` SQLite table (`docs/04` §4.1), and derive serde once core
//! adds it. File I/O here uses hand-rolled JSON so this crate is
//! dependency-free.

pub mod manager;

pub use manager::{EnvError, EnvHandle, EnvManager, EnvPin, EnvRecord};
