//! Adapter version source (`docs/04-DATA-MODEL.md` §4.1 `runtimes.version`).
//!
//! Read-only finding: neither the [`RuntimeAdapter`] trait
//! (`src/runtime/adapter.rs` — no `VERSION` const, no `version()` method)
//! nor `runtimes/registry.json` (which carries only a schema `"version": 1`,
//! no per-runtime versions) exposes an adapter version, and both are outside
//! this slice's scope. So THIS module is the version source used by
//! `list_runtimes` instead of reporting `"unknown"`:
//!
//! 1. `<data_root>/runtimes/<id>/version.txt` when an installed bundle
//!    stamps one (env-install path writes it; absent on fresh machines).
//! 2. Otherwise the app crate version suffixed ` (bundled)`: built-in
//!    adapters ship with the app and have no independent versioning yet.
//!
//! [`RuntimeAdapter`]: crate::runtime::RuntimeAdapter

use std::path::Path;

/// Resolve a display version for one adapter id. Never returns `"unknown"`.
pub fn adapter_version(data_root: &Path, id: &str) -> String {
    let stamp = data_root.join("runtimes").join(id).join("version.txt");
    if let Ok(raw) = std::fs::read_to_string(&stamp) {
        let v = raw.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    format!("{} (bundled)", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_bundled_crate_version() {
        let dir = std::env::temp_dir().join("nexora-versions-test-no-stamp");
        let v = adapter_version(&dir, "transformers");
        assert!(
            v.contains(env!("CARGO_PKG_VERSION")),
            "expected crate version fallback, got {v}"
        );
        assert!(!v.contains("unknown"), "must never report unknown");
    }

    #[test]
    fn prefers_installed_bundle_stamp() {
        let dir =
            std::env::temp_dir().join(format!("nexora-versions-stamp-{}", std::process::id()));
        let rt = dir.join("runtimes").join("transformers");
        std::fs::create_dir_all(&rt).unwrap();
        std::fs::write(rt.join("version.txt"), "4.44.2\n").unwrap();
        assert_eq!(adapter_version(&dir, "transformers"), "4.44.2");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
