//! ComfyUI integration (`docs/06` §6.8) — the advanced image/video path.
//!
//! Flow: app installs ComfyUI + required nodes, downloads models, generates
//! the workflow JSON, launches the ComfyUI server as a supervised child
//! process, executes the prompt, and retrieves the output. Needed for M13
//! advanced image/video. The server child is tracked in [`AdapterBase`] so a
//! ComfyUI crash surfaces as `E-RUNTIME-CRASHED` while the app stays alive.

use super::adapter::*;
use crate::model::manager::ModelRecord;
use std::path::{Path, PathBuf};

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "comfyui";

/// Canonical ComfyUI source (scope item 6 decision record).
pub const COMFYUI_REPO_URL: &str = "https://github.com/comfyanonymous/ComfyUI";

/// TODO-COMFYUI-URL: exact pin + fetch method still undecided — decision
/// needed from the integrator before `install()` fetches anything:
/// `git clone` (pin + update friendly, recommended) vs portable zip; which
/// release tag / commit passed the Windows py3.11 + torch-cu121 smoke test
/// (never float on `main` — ComfyUI breaks compat often); the required
/// custom-node pack list for M13 video workflows.
/// Until the pin is set, `install()`/`prepare()` fail with
/// `E-RUNTIME-NOT-INSTALLED` (never a half-fetched checkout).
/// Default port of the supervised ComfyUI server child.
pub const DEFAULT_PORT: u16 = 8188;

pub struct ComfyUIAdapter {
    base: AdapterBase,
    port: u16,
}

impl ComfyUIAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
            port: DEFAULT_PORT,
        }
    }

    pub fn with_port(data_dir: PathBuf, port: u16) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
            port,
        }
    }

    /// ComfyUI checkout dir: `runtimes/comfyui/server/`.
    pub fn server_dir(&self) -> PathBuf {
        self.base
            .data_dir()
            .join("runtimes")
            .join(ID)
            .join("server")
    }

    /// Workflow JSON for one request, staged under the server dir so the
    /// child can load it by path. Returns the staged file path.
    pub fn stage_workflow(&self, req: &InferenceRequest) -> RuntimeResult<PathBuf> {
        let dir = self.server_dir().join("workflows");
        std::fs::create_dir_all(&dir).map_err(|e| {
            RuntimeError::new(
                codes::RUNTIME_NOT_READY,
                format!("cannot stage comfyui workflow: {e}"),
                "Check free disk space and write permission on the data directory, then retry.",
            )
        })?;
        // MVP scaffold: minimal text-to-image graph. The node-graph builder
        // (M13) replaces this template with generated workflows.
        let seed = req.seed.unwrap_or(0);
        let steps: u32 = req
            .params
            .get("steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(25);
        let cfg: f64 = req
            .params
            .get("guidance")
            .and_then(|s| s.parse().ok())
            .unwrap_or(7.0);
        let workflow = serde_json::json!({
            "prompt": {
                "3": {
                    "inputs": {
                        "seed": seed,
                        "steps": steps,
                        "cfg": cfg,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0
                    },
                    "class_type": "KSampler"
                }
            }
        })
        .to_string();
        let path = dir.join(format!("{}.json", req.model.id.replace('/', "_")));
        std::fs::write(&path, workflow).map_err(|e| {
            RuntimeError::new(
                codes::RUNTIME_NOT_READY,
                format!("cannot write comfyui workflow: {e}"),
                "Check free disk space and write permission on the data directory, then retry.",
            )
        })?;
        Ok(path)
    }
}

impl RuntimeAdapter for ComfyUIAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "ComfyUI (workflows)"
    }

    fn detect(&self, _model: &ModelRecord, model_dir: &Path) -> bool {
        // Explicit workflow staged alongside the model, or an explicit
        // runtime assignment recorded by the analyzer.
        if has_file(model_dir, "workflow.json") || has_extension(model_dir, "workflow") {
            return true;
        }
        model_dir.join("workflows").is_dir()
    }

    fn install(&self) -> RuntimeResult<()> {
        let env = default_env_ref(self.base.data_dir(), ID);
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
                "Reinstall the app or restore the runtimes/comfyui/ bundle, then retry. \
                 ComfyUI itself downloads on first install (several GB).",
            ));
        }
        let launcher = if cfg!(windows) { "py" } else { "python3" };
        supervised_command(
            PathBuf::from(launcher).as_path(),
            &[
                script.to_string_lossy().to_string(),
                "--env-dir".to_string(),
                env.dir.to_string_lossy().to_string(),
                "install".to_string(),
            ],
            self.base.data_dir(),
            ID,
        )
        .map_err(|e| {
            RuntimeError::new(
                codes::ENV_INSTALL_FAILED,
                format!("comfyui env install failed: {e}"),
                "ComfyUI + node packs are a multi-GB download. Check network and disk \
                 space, then retry from Environments.",
            )
        })?;
        Ok(())
    }

    fn prepare(&self, model: &ModelRecord, model_dir: &Path) -> RuntimeResult<()> {
        let env = prepare_common(self.base.data_dir(), ID, model_dir)?;
        let _ = model;
        if !self.server_dir().join("main.py").is_file() {
            return Err(RuntimeError::new(
                codes::RUNTIME_NOT_INSTALLED,
                format!(
                    "comfyui server checkout missing: {}",
                    self.server_dir().display()
                ),
                "Press Install on the ComfyUI runtime card — the app clones the server \
                 and required nodes automatically.",
            )
            .with_runtime(ID));
        }
        let _ = env;
        Ok(())
    }

    fn run(&self, req: InferenceRequest) -> RuntimeResult<InferenceResult> {
        let env = prepare_common(self.base.data_dir(), ID, &req.model_dir)?;
        let main = self.server_dir().join("main.py");
        if !main.is_file() {
            return Err(RuntimeError::new(
                codes::RUNTIME_NOT_INSTALLED,
                "comfyui server checkout missing",
                "Press Install on the ComfyUI runtime card first.",
            ));
        }
        let workflow = self.stage_workflow(&req)?;
        // Launch the supervised server child (tracked; survives only as long
        // as needed — stop() kills it). Stdio is piped so a crash can be
        // reported with codes instead of killing the app.
        let child = std::process::Command::new(&env.python_exe)
            .arg(main)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("--workflow")
            .arg(&workflow)
            .current_dir(self.server_dir())
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                RuntimeError::new(
                    codes::CHILD_SPAWN_FAILED,
                    format!("could not launch comfyui server: {e}"),
                    "Reinstall the ComfyUI runtime from Environments, then retry.",
                )
                .with_runtime(ID)
            })?;
        // MVP scaffold: wait for completion of the single queued workflow and
        // read outputs from the output dir. Streaming progress (M13) keeps the
        // child alive across calls instead.
        let started = std::time::Instant::now();
        let output = child.wait_with_output().map_err(|e| {
            RuntimeError::crashed(format!("comfyui server wait failed: {e}"), "").with_runtime(ID)
        })?;
        if !output.status.success() {
            let tail = String::from_utf8_lossy(&output.stderr);
            let tail: String = tail
                .chars()
                .rev()
                .take(800)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            return Err(
                RuntimeError::crashed("comfyui server exited with an error", tail.trim())
                    .with_runtime(ID),
            );
        }
        let _ = self.base.stop_child();
        let out_dir = req
            .output_dir
            .clone()
            .unwrap_or_else(|| self.base.data_dir().join("outputs").join("Images"));
        let files: Vec<PathBuf> = std::fs::read_dir(&out_dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file())
                    .collect()
            })
            .unwrap_or_default();
        let mut sidecar = std::collections::HashMap::new();
        sidecar.insert("runtime".to_string(), ID.to_string());
        sidecar.insert("model".to_string(), req.model.repository.clone());
        sidecar.insert("prompt".to_string(), req.prompt.clone());
        sidecar.insert(
            "workflow".to_string(),
            workflow.to_string_lossy().to_string(),
        );
        Ok(InferenceResult {
            text: None,
            files,
            runtime_id: ID.to_string(),
            sidecar,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    fn stop(&self) -> RuntimeResult<()> {
        self.base.stop_child()
    }

    fn health_check(&self) -> RuntimeResult<()> {
        if self.base.child_running() {
            return Ok(());
        }
        if self.server_dir().join("main.py").is_file() {
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
