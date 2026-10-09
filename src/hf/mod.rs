//! Hugging Face Hub client (docs/05-HF-INTEGRATION.md).
//! Token comes from the OS environment only — never a plaintext DB column.

pub mod client;

pub use client::{
    fetch_metadata, fetch_repo_json, fetch_repo_text, hf_token_from_env, parse_hf_url,
    repo_file_url, resolve_bare_name, HfFileEntry, HfRepo, RepoMetadata,
};
