//! Storage layout + content-hash dedup + storage actions
//! (docs/04 §4.5, docs/08 §8.3–8.4).

pub mod actions;
pub mod layout;

pub use actions::{
    prune_orphan_blobs, refcount, relocate_dir, ClearReport, DeleteReport, StorageActions,
    StorageUsage,
};
pub use layout::{content_hash_hex, dedup_path, StorageLayout};
