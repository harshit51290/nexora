//! ONNX adapter, v0.5 (`docs/06` §6.5).
//!
//! Runs `.onnx` graphs without full PyTorch. Execution providers in priority
//! order: CUDA → DirectML (Windows priority for AMD/Intel/iGPU) → CPU.
//! Key path for non-NVIDIA hardware later.

use super::adapter::*;
use std::path::PathBuf;

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "onnx";

/// Execution provider preference for this machine.
///
/// Windows prioritizes DirectML (covers AMD/Intel/iGPU); elsewhere CUDA is
/// probed first and CPU is always the last-resort fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cuda,
    DirectMl,
    Cpu,
}

/// Ordered provider list to offer ONNX Runtime. `cuda_available` comes from
/// the hardware manager; DirectML is offered on Windows regardless.
pub fn provider_priority(cuda_available: bool) -> Vec<ExecutionProvider> {
    let mut providers = Vec::with_capacity(3);
    if cfg!(windows) {
        if cuda_available {
            providers.push(ExecutionProvider::Cuda);
        }
        providers.push(ExecutionProvider::DirectMl);
    } else if cuda_available {
        providers.push(ExecutionProvider::Cuda);
    }
    providers.push(ExecutionProvider::Cpu);
    providers
}

pub struct OnnxAdapter {
    base: AdapterBase,
}

impl OnnxAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
        }
    }
}

impl RuntimeAdapter for OnnxAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "ONNX Runtime (CPU / CUDA / DirectML)"
    }

    fn detect(&self, model: &Model) -> bool {
        if model.has_extension("onnx") {
            return true;
        }
        model
            .architectures
            .iter()
            .any(|a| a.eq_ignore_ascii_case("OnnxModel"))
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
            // ONNX ships in v0.5; the bundle legitimately may not exist yet.
            return Err(RuntimeError::new(
                codes::RUNTIME_NOT_INSTALLED,
                format!("runtime bundle missing (v0.5 runtime): {}", script.display()),
                "ONNX support lands in v0.5. Until then, try the Transformers or \
                 Custom Python adapter from the model page.",
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
        )
        .map_err(|e| {
            RuntimeError::new(
                codes::ENV_INSTALL_FAILED,
                format!("onnx env install failed: {e}"),
                "Check disk space and network (onnxruntime wheels are small; failure is \
                 usually a blocked PyPI mirror), then retry from Environments.",
            )
        })?;
        Ok(())
    }

    fn prepare(&self, model: &Model) -> RuntimeResult<()> {
        let _env = prepare_common(self.base.data_dir(), ID, model)?;
        if !model.has_extension("onnx") {
            return Err(RuntimeError::unsupported(
                "onnx adapter needs .onnx graph files in the model directory",
            ));
        }
        Ok(())
    }

    fn run(&self, req: InferenceRequest) -> RuntimeResult<InferenceResult> {
        let env = prepare_common(self.base.data_dir(), ID, &req.model)?;
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
        )?;
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut sidecar = std::collections::HashMap::new();
        sidecar.insert("runtime".to_string(), ID.to_string());
        sidecar.insert("model".to_string(), req.model.repository.clone());
        Ok(InferenceResult {
            text: if text.is_empty() { None } else { Some(text) },
            files: Vec::new(),
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
        let env = default_env_ref(self.base.data_dir(), ID);
        if env.python_exe.is_file() {
            Ok(())
        } else {
            Err(RuntimeError::env_missing(&env.id, &env.dir))
        }
    }
}
