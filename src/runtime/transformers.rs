//! Transformers adapter — first adapter (`docs/06` §6.2).
//!
//! Chain: isolated Python → Transformers → PyTorch → CUDA. Never touches the
//! user's global Python; everything runs from `environments/transformers/`.
//! MVP tasks: text generation (+ classification / embedding / vision
//! passthrough via the generic `params` map).

use super::adapter::*;
use std::path::PathBuf;

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "transformers";

/// `config.json → architectures` values this adapter can execute.
/// Whisper is intentionally included: the registry lists
/// `WhisperForConditionalGeneration → transformers/whisper(audio)`, so both
/// this adapter and the audio adapter may claim it; the analyzer picks via
/// HW compat (`docs/05` §5.5).
const SUPPORTED_ARCHS: &[&str] = &[
    // Causal / seq2seq LLMs (text-generation, chat).
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "FalconForCausalLM",
    "MptForCausalLM",
    "BloomForCausalLM",
    "GPTJForCausalLM",
    "GPTNeoXForCausalLM",
    "T5ForConditionalGeneration",
    "BartForConditionalGeneration",
    // Classification / embedding / reranking passthrough.
    "BertForSequenceClassification",
    "RobertaForSequenceClassification",
    "BertModel",
    "RobertaModel",
    // Vision-language passthrough.
    "LlavaForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    // Audio (shared with the audio adapter; see above).
    "WhisperForConditionalGeneration",
];

pub struct TransformersAdapter {
    base: AdapterBase,
}

impl TransformersAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
        }
    }

    /// True when the model's declared architecture is in our table.
    pub fn supports_arch(arch: &str) -> bool {
        SUPPORTED_ARCHS.contains(&arch)
    }
}

impl RuntimeAdapter for TransformersAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "Transformers (PyTorch)"
    }

    fn detect(&self, model: &Model) -> bool {
        // Explicit metadata wins (analyzer fills these from config.json).
        if model.architectures.iter().any(|a| Self::supports_arch(a)) {
            return true;
        }
        // Fall back to on-disk inspection: a HF transformers layout has
        // config.json (+ weights) but must NOT be a diffusers / GGUF / ONNX
        // layout — those belong to their own adapters.
        if model.has_file("model_index.json")
            || model.has_extension("gguf")
            || model.has_extension("onnx")
        {
            return false;
        }
        match model.read_meta("config.json") {
            Some(text) => json_string_array(&text, "architectures")
                .iter()
                .any(|a| Self::supports_arch(a)),
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
                "Reinstall the app or restore the runtimes/transformers/ bundle, then retry.",
            ));
        }
        // Bootstrap needs *an* interpreter to create the venv; the system
        // Python is used only as a launcher — packages always go into the
        // isolated env, never global pip.
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
                format!("transformers env install failed: {e}"),
                "Check disk space and network (PyTorch wheels are ~2GB), then retry \
                 from Environments. A proxy or offline machine needs a manual wheel import.",
            )
        })?;
        Ok(())
    }

    fn prepare(&self, model: &Model) -> RuntimeResult<()> {
        let _env = prepare_common(self.base.data_dir(), ID, model)?;
        if !model.architectures.is_empty()
            && !model.architectures.iter().any(|a| Self::supports_arch(a))
            && !model.has_file("config.json")
        {
            return Err(RuntimeError::unsupported(format!(
                "transformers adapter cannot run architectures {:?}",
                model.architectures
            )));
        }
        Ok(())
    }

    fn run(&self, req: InferenceRequest) -> RuntimeResult<InferenceResult> {
        let env = prepare_common(self.base.data_dir(), ID, &req.model)?;
        let entry = require_serve_entrypoint(self.base.data_dir(), ID, &env)?;
        // One-shot executor until the persistent server lands: spawn
        // `serve.py --job <json>`, wait, and map crashes to error codes.
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
        // One-shot executor: nothing persistent to stop (yet).
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
