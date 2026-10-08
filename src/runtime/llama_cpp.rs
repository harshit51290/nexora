//! llama.cpp adapter for GGUF weights (`docs/06` §6.4).
//!
//! GGUF jargon stays hidden from the UI: the adapter reports Quantization
//! (Q4_K_M default on 4 GB), estimated RAM, GPU offload layers, and expected
//! performance. The recommendation engine ([`recommend_quant`]) picks the
//! Q8/Q6/Q5/Q4/Q3 variant balancing VRAM against quality.

use super::adapter::*;
use std::path::PathBuf;

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "llama_cpp";

pub struct LlamaCppAdapter {
    base: AdapterBase,
}

impl LlamaCppAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
        }
    }

    /// Path of the native server binary managed by this adapter, once
    /// installed (shipped per-OS; never built from source by the app).
    pub fn server_binary(&self) -> PathBuf {
        let exe = if cfg!(windows) {
            "llama-server.exe"
        } else {
            "llama-server"
        };
        self.base
            .data_dir()
            .join("runtimes")
            .join(ID)
            .join("bin")
            .join(exe)
    }
}

/// Pick the quantization variant balancing VRAM against quality.
///
/// * 4 GB reference machine: `Q4_K_M` default (fits ~7-8B with offload).
/// * Larger files on small VRAM step down Q8 → Q6 → Q5 → Q4 → Q3.
/// * Plenty of VRAM relative to size: stay at Q8 for max quality.
pub fn recommend_quant(model_size_gb: f32, vram_gb: f32) -> &'static str {
    let ratio = if model_size_gb > 0.0 {
        vram_gb / model_size_gb
    } else {
        1.0
    };
    if ratio >= 1.6 {
        "Q8_0"
    } else if ratio >= 1.1 {
        "Q6_K"
    } else if ratio >= 0.85 {
        "Q5_K_M"
    } else if ratio >= 0.55 {
        "Q4_K_M"
    } else {
        "Q3_K_M"
    }
}

/// GPU offload layers for a VRAM budget: keep ~0.5 GB headroom for context.
pub fn offload_layers(total_layers: u32, vram_gb: f32, bytes_per_layer_gb: f32) -> u32 {
    if bytes_per_layer_gb <= 0.0 {
        return 0;
    }
    let budget = (vram_gb - 0.5).max(0.0);
    let fit = (budget / bytes_per_layer_gb).floor() as u32;
    fit.min(total_layers)
}

impl RuntimeAdapter for LlamaCppAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "llama.cpp (GGUF, quantized)"
    }

    fn detect(&self, model: &Model) -> bool {
        // GGUF marker wins over everything: any .gguf weight file, or an
        // analyzer-declared GGUF format.
        if model.has_extension("gguf") {
            return true;
        }
        model
            .capabilities
            .iter()
            .any(|c| c.eq_ignore_ascii_case("gguf"))
            || model
                .architectures
                .iter()
                .any(|a| a.eq_ignore_ascii_case("GGUF"))
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
                "Reinstall the app or restore the runtimes/llama_cpp/ bundle, then retry.",
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
                format!("llama.cpp env install failed: {e}"),
                "The native server binary downloads from GitHub releases — check network \
                 access, then retry from Environments.",
            )
        })?;
        Ok(())
    }

    fn prepare(&self, model: &Model) -> RuntimeResult<()> {
        let _env = prepare_common(self.base.data_dir(), ID, model)?;
        if !model.has_extension("gguf") {
            return Err(RuntimeError::unsupported(
                "llama.cpp adapter needs .gguf weight files in the model directory",
            ));
        }
        Ok(())
    }

    fn run(&self, req: InferenceRequest) -> RuntimeResult<InferenceResult> {
        let env = prepare_common(self.base.data_dir(), ID, &req.model)?;
        // Prefer the native server binary; fall back to the Python shim
        // entrypoint once serve.py lands.
        let binary = self.server_binary();
        if !binary.is_file() {
            let _ = require_serve_entrypoint(self.base.data_dir(), ID, &env)?;
        }
        let job = job_json(ID, &req);
        let started = std::time::Instant::now();
        let (program, extra) = if binary.is_file() {
            (binary, vec!["--job".to_string(), job])
        } else {
            // Unreachable today (require_serve_entrypoint errors first) but
            // keeps the routing explicit for when serve.py lands.
            return Err(RuntimeError::new(
                codes::RUNTIME_NOT_READY,
                "llama.cpp server binary and serve.py are both missing",
                "Reinstall the llama.cpp runtime from Environments, then retry.",
            ));
        };
        let output = supervised_command(&program, &extra, self.base.data_dir())?;
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut sidecar = std::collections::HashMap::new();
        sidecar.insert("runtime".to_string(), ID.to_string());
        sidecar.insert("model".to_string(), req.model.repository.clone());
        sidecar.insert("prompt".to_string(), req.prompt.clone());
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
        if self.server_binary().is_file() {
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
