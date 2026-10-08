//! Nexora — universal local AI runtime manager.
//!
//! Orchestration-only core: model/hardware/hf/download/analyzer/storage/
//! security/scheduler modules plus the persisted [`core`] state machines and
//! error catalog. Rust coordinates; child-process runtimes execute.
//! API/CLI/Tauri layers call the same core fns; endpoints not yet wired to
//! managers fail with a coded stub error (search `TODO-WIRE`).

pub mod analyze;
pub mod api;
pub mod cli;
pub mod core;
pub mod download;
pub mod env;
pub mod hardware;
pub mod hf;
pub mod jobs;
pub mod mem;
pub mod model;
pub mod plugins;
pub mod runtime;
pub mod scheduler;
pub mod security;
pub mod storage;
