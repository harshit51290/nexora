//! Hugging Face Hub client (docs/05-HF-INTEGRATION.md).
//! Token comes from the OS environment only — never a plaintext DB column.

pub mod client;

pub use client::{
    fetch_metadata, hf_token_from_env, parse_hf_url, HfFileEntry, HfRepo, RepoMetadata,
};
