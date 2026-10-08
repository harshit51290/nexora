//! Storage layout + content-hash dedup (docs/04 §4.5, docs/08 §8.3–8.4).

pub mod layout;

pub use layout::{content_hash_hex, dedup_path, StorageLayout};
