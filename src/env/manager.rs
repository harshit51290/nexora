//! Env lifecycle: create / shared-reuse / pin / rollback.
//!
//! MVP starts 1-env-per-runtime-kind (`docs/09` §9.1); the resolver already
//! supports compatible sharing so later milestones don't need a rewrite.
//!
//! Pin records are `serde`-serialized ([`EnvRecord`]); every [`EnvError`]
//! converts into the core [`crate::core::NexoraError`] via `From` with the
//! stable code + human fix embedded (see NEED-CORE-ERRORS in
//! `src/runtime/adapter.rs` — the core catalog has no env-specific variants
//! yet).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::core::NexoraError;

/// Stable error codes for env operations (see the core-mapping note above).
pub mod codes {
    pub const ENV_MISSING: &str = "E-ENV-MISSING";
    pub const ENV_CREATE_FAILED: &str = "E-ENV-CREATE-FAILED";
    pub const PIN_INCOMPATIBLE: &str = "E-PIN-INCOMPATIBLE";
    pub const ROLLBACK_MISSING: &str = "E-ROLLBACK-MISSING";
}

#[derive(Debug, Clone)]
pub struct EnvError {
    pub code: &'static str,
    pub message: String,
    pub hint: String,
}

impl EnvError {
    pub fn new(code: &'static str, message: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            hint: hint.into(),
        }
    }

    /// The stable wire code. Never changes; safe to persist and match on.
    pub fn stable_code(&self) -> &'static str {
        self.code
    }
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} Fix: {}", self.code, self.message, self.hint)
    }
}

impl std::error::Error for EnvError {}

/// Convert into the core catalog. All env codes travel inside `Other` with
/// the stable code + human fix embedded (`[E-...] … Fix: …`); narrow this
/// once `src/core/error.rs` gains env variants (NEED-CORE-ERRORS).
impl From<EnvError> for NexoraError {
    fn from(e: EnvError) -> Self {
        NexoraError::Other(anyhow::anyhow!(
            "[{}] {} Fix: {}",
            e.code,
            e.message,
            e.hint
        ))
    }
}

pub type EnvResult<T> = Result<T, EnvError>;

/// Reproducibility pin, e.g. `Env#17: py3.11/torch2.x/transf5.x/cu12.x`
/// (`docs/09` §9.2). Model updates are optional and never silent, so the
/// model revision is part of the pin.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvPin {
    /// `major.minor`, e.g. `"3.11"`.
    #[serde(default)]
    pub python: String,
    /// Package → pinned version, e.g. `{"torch": "2.3.1", "transformers": "4.44.2"}`.
    #[serde(default)]
    pub packages: HashMap<String, String>,
    /// CUDA build, e.g. `"cu121"`, or `"cpu"` / `"directml"`.
    #[serde(default)]
    pub cuda: String,
    /// Pinned model `@rev`, if the env is model-specific.
    #[serde(default)]
    pub model_rev: Option<String>,
}

impl EnvPin {
    /// Compatible when interpreter minor and CUDA build match and every
    /// already-pinned package required by `other` agrees. Extra packages on
    /// either side are fine (superset envs are reusable).
    pub fn compatible_with(&self, other: &EnvPin) -> bool {
        if self.python != other.python || self.cuda != other.cuda {
            return false;
        }
        other
            .packages
            .iter()
            .all(|(name, ver)| match self.packages.get(name) {
                Some(mine) => mine == ver,
                None => true,
            })
    }
}

/// What the manager hands out: resolved paths, never global interpreters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvHandle {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub dir: PathBuf,
    #[serde(default)]
    pub python_exe: PathBuf,
    #[serde(default)]
    pub environment_json: PathBuf,
}

/// Persisted env record (`environment.json` next to the env).
///
/// `#[serde(default)]` on every field plus default unknown-field tolerance
/// keeps forward/backward reads working: stub files carry an extra `note`,
/// and the llama.cpp pin carries an extra `server_tag` — both ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub pin: EnvPin,
    /// `ready | installing | broken`.
    #[serde(default)]
    pub status: String,
    /// Unix seconds.
    #[serde(default)]
    pub created_at: u64,
}

impl EnvRecord {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| "{\"status\":\"broken\"}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        serde_json::from_str(text).ok()
    }
}

fn now_secs() -> u64 {
    chrono::Utc::now().timestamp().max(0) as u64
}

/// Manages `environments/` under the relocatable data root.
pub struct EnvManager {
    data_root: PathBuf,
}

impl EnvManager {
    pub fn new(data_root: PathBuf) -> Self {
        Self { data_root }
    }

    pub fn envs_root(&self) -> PathBuf {
        self.data_root.join("environments")
    }

    pub fn env_dir(&self, id: &str) -> PathBuf {
        self.envs_root().join(id)
    }

    pub fn environment_json(&self, id: &str) -> PathBuf {
        self.env_dir(id).join("environment.json")
    }

    /// Previous pin, kept for one-click rollback (`docs/09` §9.2).
    pub fn prev_json(&self, id: &str) -> PathBuf {
        self.env_dir(id).join("environment.prev.json")
    }

    /// MVP default: 1-env-per-runtime-kind → the adapter convention
    /// `environments/<kind>` (matches `bootstrap.py --env-dir` and
    /// `runtime::adapter::default_env_ref`). Collisions mint `<kind>-N`.
    pub fn default_env_id(kind: &str) -> String {
        kind.to_string()
    }

    #[cfg(windows)]
    pub fn python_exe(&self, id: &str) -> PathBuf {
        self.env_dir(id)
            .join("python")
            .join("Scripts")
            .join("python.exe")
    }

    #[cfg(not(windows))]
    pub fn python_exe(&self, id: &str) -> PathBuf {
        self.env_dir(id).join("python").join("bin").join("python")
    }

    /// Ready = pin file + isolated interpreter both present.
    pub fn is_ready(&self, id: &str) -> bool {
        self.environment_json(id).is_file() && self.python_exe(id).is_file()
    }

    pub fn handle(&self, id: &str, kind: &str) -> EnvHandle {
        EnvHandle {
            python_exe: self.python_exe(id),
            environment_json: self.environment_json(id),
            id: id.to_string(),
            kind: kind.to_string(),
            dir: self.env_dir(id),
        }
    }

    /// Read a persisted pin record. `None` when the file is missing or
    /// unparseable — callers treat that as "never installed", never fatal.
    pub fn read_record(&self, id: &str) -> Option<EnvRecord> {
        let text = std::fs::read_to_string(self.environment_json(id)).ok()?;
        EnvRecord::from_json(&text)
    }

    /// Pin only (see [`EnvManager::read_record`]).
    pub fn read_pin(&self, id: &str) -> Option<EnvPin> {
        self.read_record(id).map(|r| r.pin)
    }

    /// Resolve the env a runtime kind should use (`docs/09` §9.1).
    ///
    /// When the conventional env (`environments/<kind>`) has a recorded pin,
    /// that pin is the sharing key: an existing *compatible* env (same
    /// interpreter + CUDA + agreed packages, [`EnvPin::compatible_with`]) is
    /// reused (`A+B+C → one Transformers env`); otherwise a fresh
    /// `<kind>[-N]` id is minted for creation. When no pin was ever
    /// recorded (never installed), the conventional path is returned
    /// directly so the caller fails with `E-ENV-MISSING` → Install prompt.
    /// Unreadable pin files are skipped, never fatal.
    pub fn resolve_for_runtime(&self, kind: &str) -> EnvHandle {
        match self.read_pin(kind) {
            Some(want) => self.resolve_shared_env(kind, &want),
            None => self.handle(kind, kind),
        }
    }

    /// Shared-reuse resolver (`docs/09` §9.1): scan existing envs of this
    /// kind for a compatible pin and reuse it (`A+B+C → one Transformers
    /// env`); otherwise mint a fresh `<kind>[-N]` id for creation.
    /// Unreadable pin files are skipped, never fatal.
    pub fn resolve_shared_env(&self, kind: &str, want: &EnvPin) -> EnvHandle {
        let root = self.envs_root();
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                let id = match dir.file_name().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                let record = match self.read_record(&id) {
                    Some(r) => r,
                    None => continue,
                };
                if record.kind != kind || record.status != "ready" {
                    continue;
                }
                if !record.pin.compatible_with(want) {
                    continue;
                }
                if !self.python_exe(&id).is_file() {
                    continue;
                }
                return self.handle(&id, kind);
            }
        }
        // No compatible env — mint a fresh id.
        let base = Self::default_env_id(kind);
        if !self.env_dir(&base).exists() {
            return self.handle(&base, kind);
        }
        let mut n = 2u32;
        loop {
            let id = format!("{base}-{n}");
            if !self.env_dir(&id).exists() {
                return self.handle(&id, kind);
            }
            n += 1;
        }
    }

    /// Write a new pin, backing up the current `environment.json` first so
    /// [`EnvManager::rollback`] can restore it. A runtime update that breaks
    /// a model is therefore always one click away from the last good env.
    pub fn record_pin(&self, id: &str, kind: &str, pin: &EnvPin) -> EnvResult<()> {
        let dir = self.env_dir(id);
        std::fs::create_dir_all(&dir).map_err(|e| {
            EnvError::new(
                codes::ENV_CREATE_FAILED,
                format!("cannot create env dir {}: {e}", dir.display()),
                "Check free disk space and write permission on the data directory, then retry.",
            )
        })?;
        let current = self.environment_json(id);
        if current.is_file() {
            std::fs::copy(&current, self.prev_json(id)).map_err(|e| {
                EnvError::new(
                    codes::ENV_CREATE_FAILED,
                    format!("cannot back up env pin: {e}"),
                    "Check write permission on the environments directory, then retry.",
                )
            })?;
        }
        let record = EnvRecord {
            id: id.to_string(),
            kind: kind.to_string(),
            pin: pin.clone(),
            status: "ready".to_string(),
            created_at: now_secs(),
        };
        std::fs::write(&current, record.to_json()).map_err(|e| {
            EnvError::new(
                codes::ENV_CREATE_FAILED,
                format!("cannot write env pin: {e}"),
                "Check free disk space and write permission on the data directory, then retry.",
            )
        })?;
        Ok(())
    }

    /// Restore the pre-update pin. Fails with `E-ROLLBACK-MISSING` (not bare
    /// I/O noise) when no backup exists.
    pub fn rollback(&self, id: &str) -> EnvResult<()> {
        let prev = self.prev_json(id);
        if !prev.is_file() {
            return Err(EnvError::new(
                codes::ROLLBACK_MISSING,
                format!("no previous pin kept for env '{id}'"),
                "Nothing to roll back to — this env was never updated. Reinstall it from \
                 Environments if it is broken.",
            ));
        }
        std::fs::copy(&prev, self.environment_json(id)).map_err(|e| {
            EnvError::new(
                codes::ENV_CREATE_FAILED,
                format!("rollback copy failed: {e}"),
                "Check write permission on the environments directory, then retry.",
            )
        })?;
        Ok(())
    }
}

/// Interpreter path helper for non-manager contexts (mirrors
/// `runtime::adapter::venv_python`).
pub fn venv_python(env_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        env_dir.join("python").join("Scripts").join("python.exe")
    } else {
        env_dir.join("python").join("bin").join("python")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_record_tolerates_stub_extras() {
        // llama.cpp stub shape: extra top-level `note`, extra `server_tag`
        // inside the pin. Both must be ignored, not fatal.
        let raw = r#"{"id":"env-llama_cpp","kind":"llama_cpp","status":"ready",
            "created_at":0,"note":"STUB pin",
            "pin":{"python":"3.11","cuda":"cpu","model_rev":null,
            "server_tag":null,"packages":{}}}"#;
        let rec = EnvRecord::from_json(raw).expect("stub parses");
        assert_eq!(rec.kind, "llama_cpp");
        assert_eq!(rec.pin.python, "3.11");
        let back = rec.to_json();
        let again = EnvRecord::from_json(&back).expect("round-trips");
        assert_eq!(again.pin.cuda, "cpu");
    }

    #[test]
    fn default_env_id_matches_adapter_convention() {
        // Adapters + bootstrap.py use environments/<kind>.
        assert_eq!(EnvManager::default_env_id("transformers"), "transformers");
        assert_eq!(EnvManager::default_env_id("llama_cpp"), "llama_cpp");
    }

    #[test]
    fn env_error_maps_with_code_and_fix() {
        let e = EnvError::new(
            codes::ROLLBACK_MISSING,
            "no previous pin kept for env 'transformers'",
            "Nothing to roll back to.",
        );
        assert_eq!(e.stable_code(), codes::ROLLBACK_MISSING);
        let n = NexoraError::from(e);
        let shown = format!("{n}");
        assert!(shown.contains("E-ROLLBACK-MISSING"));
        assert!(!n.human_fix().is_empty());
    }

    #[test]
    fn shared_resolver_reuses_compatible_env() {
        let root = std::env::temp_dir().join(format!("nexora-env-test-{}", now_secs()));
        let _ = std::fs::remove_dir_all(&root);
        let mgr = EnvManager::new(root.clone());
        let mut packages = HashMap::new();
        packages.insert("torch".to_string(), "2.3.1".to_string());
        let pin = EnvPin {
            python: "3.11".to_string(),
            packages,
            cuda: "cu121".to_string(),
            model_rev: None,
        };
        mgr.record_pin("transformers", "transformers", &pin)
            .expect("record");
        // Superset pin on disk still satisfies a smaller want.
        let mut want_packages = HashMap::new();
        want_packages.insert("torch".to_string(), "2.3.1".to_string());
        let want = EnvPin {
            python: "3.11".to_string(),
            packages: want_packages,
            cuda: "cu121".to_string(),
            model_rev: None,
        };
        assert!(pin.compatible_with(&want));
        // No interpreter present → not reusable; a fresh id is minted.
        let handle = mgr.resolve_shared_env("transformers", &want);
        assert_eq!(handle.id, "transformers-2");
        let _ = std::fs::remove_dir_all(&root);
    }
}
