//! Env lifecycle: create / shared-reuse / pin / rollback.
//!
//! MVP starts 1-env-per-runtime-kind (`docs/09` §9.1); the resolver already
//! supports compatible sharing so later milestones don't need a rewrite.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Stable error codes for env operations (mirrors the runtime catalog;
/// TODO-CORE-ALIGN: merge into one core catalog).
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
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} Fix: {}", self.code, self.message, self.hint)
    }
}

impl std::error::Error for EnvError {}

pub type EnvResult<T> = Result<T, EnvError>;

/// Reproducibility pin, e.g. `Env#17: py3.11/torch2.x/transf5.x/cu12.x`
/// (`docs/09` §9.2). Model updates are optional and never silent, so the
/// model revision is part of the pin.
#[derive(Debug, Clone, Default)]
pub struct EnvPin {
    /// `major.minor`, e.g. `"3.11"`.
    pub python: String,
    /// Package → pinned version, e.g. `{"torch": "2.3.1", "transformers": "4.44.2"}`.
    pub packages: HashMap<String, String>,
    /// CUDA build, e.g. `"cu121"`, or `"cpu"` / `"directml"`.
    pub cuda: String,
    /// Pinned model `@rev`, if the env is model-specific.
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
#[derive(Debug, Clone)]
pub struct EnvHandle {
    pub id: String,
    pub kind: String,
    pub dir: PathBuf,
    pub python_exe: PathBuf,
    pub environment_json: PathBuf,
}

/// Persisted env record (`environment.json` next to the env).
#[derive(Debug, Clone)]
pub struct EnvRecord {
    pub id: String,
    pub kind: String,
    pub pin: EnvPin,
    /// `ready | installing | broken`.
    pub status: String,
    /// Unix seconds, via `std::time` (TODO-CORE-ALIGN: chrono).
    pub created_at: u64,
}

impl EnvRecord {
    pub fn to_json(&self) -> String {
        let mut pkgs: Vec<String> = self
            .pin
            .packages
            .iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", escape(k), escape(v)))
            .collect();
        pkgs.sort();
        format!(
            "{{\"id\":\"{}\",\"kind\":\"{}\",\"status\":\"{}\",\"created_at\":{},\
             \"pin\":{{\"python\":\"{}\",\"cuda\":\"{}\",\"model_rev\":{},\"packages\":{{{}}}}}}}",
            escape(&self.id),
            escape(&self.kind),
            escape(&self.status),
            self.created_at,
            escape(&self.pin.python),
            escape(&self.pin.cuda),
            match &self.pin.model_rev {
                Some(rev) => format!("\"{}\"", escape(rev)),
                None => "null".to_string(),
            },
            pkgs.join(","),
        )
    }
}

fn escape(raw: &str) -> String {
    raw.replace('\\', "\\\\").replace('"', "\\\"")
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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

    /// MVP default: 1-env-per-runtime-kind → `env-<kind>`.
    pub fn default_env_id(kind: &str) -> String {
        format!("env-{kind}")
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

    /// Shared-reuse resolver (`docs/09` §9.1): scan existing envs of this
    /// kind for a compatible pin and reuse it (`A+B+C → one Transformers
    /// env`); otherwise mint a fresh `env-<kind>[-N]` id for creation.
    /// Unreadable pin files are skipped, never fatal.
    pub fn resolve_shared_env(&self, kind: &str, want: &EnvPin) -> EnvHandle {
        let root = self.envs_root();
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                let record_file = dir.join("environment.json");
                let text = match std::fs::read_to_string(&record_file) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if extract(&text, "\"kind\"") != Some(kind.to_string()) {
                    continue;
                }
                if extract(&text, "\"status\"") != Some("ready".to_string()) {
                    continue;
                }
                let have = EnvPin {
                    python: extract(&text, "\"python\"").unwrap_or_default(),
                    cuda: extract(&text, "\"cuda\"").unwrap_or_default(),
                    model_rev: None,
                    packages: HashMap::new(),
                };
                if have.python == want.python
                    && have.cuda == want.cuda
                    && self.python_exe(dir.file_name().and_then(|s| s.to_str()).unwrap_or("")).is_file()
                {
                    let id = dir.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
                    return self.handle(&id, kind);
                }
            }
        }
        // No compatible env — mint a fresh id.
        let base = Self::default_env_id(kind);
        if !self.env_dir(&base).exists() {
            return self.handle(&base, kind);
        }
        let mut n = 2;
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

/// Naive `"key": "value"` scan for pin files (dependency-free).
/// TODO-CORE-ALIGN: replace with serde_json.
fn extract(text: &str, key: &str) -> Option<String> {
    let i = text.find(key)?;
    let after = text[i + key.len()..].trim_start();
    let after = after.strip_prefix(':')?.trim_start();
    let quoted = after.strip_prefix('"')?;
    let end = quoted.find('"')?;
    Some(quoted[..end].to_string())
}

/// Interpreter path helper for non-manager contexts (mirrors
/// `runtime::adapter::venv_python`; TODO-CORE-ALIGN: keep one).
pub fn venv_python(env_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        env_dir.join("python").join("Scripts").join("python.exe")
    } else {
        env_dir.join("python").join("bin").join("python")
    }
}
