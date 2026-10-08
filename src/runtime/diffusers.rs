//! Diffusers adapter (`docs/06` §6.3).
//!
//! Handles `model_index.json` pipelines: SD / SDXL / Flux + image editing.
//! Low-VRAM flags (FP16, attention slicing, VAE tiling, CPU offload) are
//! auto-applied from the hardware profile — see [`low_vram_flags`].

use super::adapter::*;
use std::path::PathBuf;

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "diffusers";

/// `_class_name` values (from `model_index.json`) this adapter executes.
const SUPPORTED_PIPELINES: &[&str] = &[
    "StableDiffusionPipeline",
    "StableDiffusionImg2ImgPipeline",
    "StableDiffusionInpaintPipeline",
    "StableDiffusionXLPipeline",
    "StableDiffusionXLImg2ImgPipeline",
    "StableDiffusionXLInpaintPipeline",
    "StableDiffusion3Pipeline",
    "FluxPipeline",
    "KandinskyPipeline",
];

pub struct DiffusersAdapter {
    base: AdapterBase,
}

impl DiffusersAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
        }
    }

    pub fn supports_pipeline(class_name: &str) -> bool {
        SUPPORTED_PIPELINES.contains(&class_name)
    }
}

/// Memory-saver flags auto-applied per hardware profile.
///
/// * ≤ 4 GB (reference low-end): everything on — FP16, attention slicing,
///   VAE slicing + tiling, sequential CPU offload.
/// * ≤ 8 GB: FP16 + slicing + model CPU offload, no tiling.
/// * above: FP16 only (full GPU).
pub fn low_vram_flags(vram_gb: f32) -> Vec<String> {
    if vram_gb <= 4.0 {
        vec![
            "fp16".to_string(),
            "attention_slicing".to_string(),
            "vae_slicing".to_string(),
            "vae_tiling".to_string(),
            "sequential_cpu_offload".to_string(),
        ]
    } else if vram_gb <= 8.0 {
        vec![
            "fp16".to_string(),
            "attention_slicing".to_string(),
            "model_cpu_offload".to_string(),
        ]
    } else {
        vec!["fp16".to_string()]
    }
}

impl RuntimeAdapter for DiffusersAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "Diffusers (image generation)"
    }

    fn detect(&self, model: &Model) -> bool {
        // Analyzer-declared pipeline class wins.
        if model.architectures.iter().any(|a| Self::supports_pipeline(a)) {
            return true;
        }
        // On-disk: presence of model_index.json with a known _class_name.
        match model.read_meta("model_index.json") {
            Some(text) => json_string(&text, "_class_name")
                .map(|class| Self::supports_pipeline(&class))
                .unwrap_or(false),
            None => false,
        }
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
                "Reinstall the app or restore the runtimes/diffusers/ bundle, then retry.",
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
                format!("diffusers env install failed: {e}"),
                "Check disk space and network (PyTorch + diffusers wheels are ~2-3GB), \
                 then retry from Environments.",
            )
        })?;
        Ok(())
    }

    fn prepare(&self, model: &Model) -> RuntimeResult<()> {
        let _env = prepare_common(self.base.data_dir(), ID, model)?;
        if !model.has_file("model_index.json")
            && !model
                .architectures
                .iter()
                .any(|a| Self::supports_pipeline(a))
        {
            return Err(RuntimeError::unsupported(
                "diffusers adapter needs a model_index.json pipeline (SD/SDXL/Flux layout)",
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
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // Convention: serve.py prints the output image path on its last line.
        let files: Vec<PathBuf> = stdout
            .lines()
            .last()
            .map(|line| {
                let p = PathBuf::from(line.trim());
                if p.is_file() {
                    vec![p]
                } else {
                    Vec::new()
                }
            })
            .unwrap_or_default();
        let mut sidecar = std::collections::HashMap::new();
        sidecar.insert("runtime".to_string(), ID.to_string());
        sidecar.insert("model".to_string(), req.model.repository.clone());
        sidecar.insert("prompt".to_string(), req.prompt.clone());
        if let Some(seed) = req.seed {
            sidecar.insert("seed".to_string(), seed.to_string());
        }
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
        let env = default_env_ref(self.base.data_dir(), ID);
        if env.python_exe.is_file() {
            Ok(())
        } else {
            Err(RuntimeError::env_missing(&env.id, &env.dir))
        }
    }
}
