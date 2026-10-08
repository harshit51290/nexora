//! Custom / experimental Python adapter (`docs/05` §5.6, `docs/06` §6.9).
//!
//! Unsupported architectures land here behind **explicit user consent**:
//! detecting `requirements.txt` / `pyproject.toml` / `setup.py` /
//! `environment.yml` / `custom_model.py` proposes an isolated
//! Python+Torch+Transformers/Accelerate env, but `prepare`/`run` refuse with
//! [`codes::CUSTOM_CODE_BLOCKED`] until the user picks View / Sandbox /
//! Cancel in the trust modal (`docs/09-ENV-SECURITY.md` §9.4). Consent is
//! never silent and never persisted beyond the session by this adapter.

use super::adapter::*;
use crate::model::manager::ModelRecord;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "custom";

/// Dependency manifests that mark a repo as custom-code runnable.
const CUSTOM_MARKERS: &[&str] = &[
    "custom_model.py",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "environment.yml",
];

pub struct CustomAdapter {
    base: AdapterBase,
    /// Explicit per-session consent from the trust modal. Defaults to false.
    consented: AtomicBool,
}

impl CustomAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
            consented: AtomicBool::new(false),
        }
    }

    /// Record the trust-modal decision for this session. Called by the
    /// security manager — never set implicitly by detection or download.
    pub fn set_consent(&self, granted: bool) {
        self.consented.store(granted, Ordering::SeqCst);
    }

    fn require_consent(&self, model: &ModelRecord, model_dir: &Path) -> RuntimeResult<()> {
        if self.consented.load(Ordering::SeqCst) {
            return Ok(());
        }
        let markers: Vec<&str> = CUSTOM_MARKERS
            .iter()
            .copied()
            .filter(|m| has_file(model_dir, m))
            .collect();
        Err(RuntimeError::new(
            codes::CUSTOM_CODE_BLOCKED,
            format!(
                "model '{}' ships custom code ({}) and needs explicit consent",
                model.repository,
                if markers.is_empty() {
                    "unrecognized layout".to_string()
                } else {
                    markers.join(", ")
                }
            ),
            "Open the trust prompt: [View Files] to inspect the code, [Run in Sandbox] \
             to consent for this session, or [Cancel]. The app never executes repo code blindly.",
        ))
    }
}

impl RuntimeAdapter for CustomAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "Custom Python (experimental)"
    }

    fn detect(&self, model: &ModelRecord, model_dir: &Path) -> bool {
        // Catch-all for repo-code layouts. Deliberately claims anything with
        // custom-code markers OR an unrecognized architecture — the trust gate
        // (not detection) is what protects the user.
        if CUSTOM_MARKERS.iter().any(|m| has_file(model_dir, m)) {
            return true;
        }
        if model.trust_level.as_deref() == Some("Blocked") {
            return false;
        }
        // Unrecognized arch with a config.json: offer experimental, don't crash.
        has_file(model_dir, "config.json") && !config_architectures(model_dir).is_empty()
    }

    fn install(&self) -> RuntimeResult<()> {
        // The env is created per-model on consented prepare(), not here:
        // requirements differ per repo. install() only verifies the bundle.
        let script = self
            .base
            .data_dir()
            .join("runtimes")
            .join(ID)
            .join("bootstrap.py");
        if !script.is_file() {
            return Err(RuntimeError::new(
                codes::RUNTIME_NOT_INSTALLED,
                format!("runtime bundle missing: {}", script.display()),
                "Reinstall the app or restore the runtimes/custom/ bundle, then retry.",
            ));
        }
        Ok(())
    }

    fn prepare(&self, model: &ModelRecord, model_dir: &Path) -> RuntimeResult<()> {
        self.require_consent(model, model_dir)?;
        let _env = prepare_common(self.base.data_dir(), ID, model_dir)?;
        Ok(())
    }

    fn run(&self, req: InferenceRequest) -> RuntimeResult<InferenceResult> {
        self.require_consent(&req.model, &req.model_dir)?;
        let env = prepare_common(self.base.data_dir(), ID, &req.model_dir)?;
        let entry = require_serve_entrypoint(self.base.data_dir(), ID, &env)?;
        let job = job_json(ID, &req);
        let started = std::time::Instant::now();
        let output = supervised_command(
            &env.python_exe,
            &[
                entry.to_string_lossy().to_string(),
                "--job".to_string(),
                job,
            ],
            self.base.data_dir(),
            ID,
        )?;
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut sidecar = std::collections::HashMap::new();
        sidecar.insert("runtime".to_string(), ID.to_string());
        sidecar.insert("model".to_string(), req.model.repository.clone());
        sidecar.insert("experimental".to_string(), "true".to_string());
        Ok(InferenceResult {
            text: if text.is_empty() { None } else { Some(text) },
            files: Vec::new(),
            runtime_id: ID.to_string(),
            sidecar,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn stop(&self) -> RuntimeResult<()> {
        // Consent lapses on stop: a new run needs a fresh trust decision.
        self.consented.store(false, Ordering::SeqCst);
        self.base.stop_child()
    }

    fn health_check(&self) -> RuntimeResult<()> {
        if self.base.child_running() {
            return Ok(());
        }
        let env = default_env_ref(self.base.data_dir(), ID);
        if env.python_exe.is_file() {
            Ok(())
        } else {
            Err(RuntimeError::env_missing(&env.id, &env.dir))
        }
    }
}
