//! Audio adapter layer (`docs/06` §6.6).
//!
//! No single engine is assumed: routing maps model families to tasks —
//! Whisper → speech-to-text, Kokoro → text-to-speech, etc. v0.3 ships
//! Whisper STT + TTS + playback; music / separation route here later.

use super::adapter::*;
use std::path::PathBuf;

/// Registry id shared with `runtimes/registry.json` and the env layout.
pub const ID: &str = "audio";

/// Audio task this model serves, derived from arch / pipeline / capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioTask {
    SpeechToText,
    TextToSpeech,
    MusicGeneration,
    SoundGeneration,
    AudioProcessing,
    Unknown,
}

const STT_ARCHS: &[&str] = &[
    "WhisperForConditionalGeneration",
    "Wav2Vec2ForCTC",
    "HuBERTForCTC",
];

const TTS_ARCHS: &[&str] = &[
    "KokoroForConditionalGeneration",
    "BarkForConditionalGeneration",
    "VitsModel",
    "SpeechT5ForTextToSpeech",
];

/// Route a model to its audio task. Analyzer metadata first, on-disk
/// markers second (`preprocessor_config.json` + whisper tokenizer for STT,
/// `voices/` or `voice*.bin` for Kokoro-style TTS).
pub fn classify_task(model: &Model) -> AudioTask {
    if model.architectures.iter().any(|a| STT_ARCHS.contains(&a.as_str())) {
        return AudioTask::SpeechToText;
    }
    if model.architectures.iter().any(|a| TTS_ARCHS.contains(&a.as_str())) {
        return AudioTask::TextToSpeech;
    }
    if let Some(tag) = model.pipeline_tag.as_deref() {
        match tag {
            "automatic-speech-recognition" => return AudioTask::SpeechToText,
            "text-to-speech" => return AudioTask::TextToSpeech,
            "text-to-audio" => return AudioTask::SoundGeneration,
            "audio-classification" => return AudioTask::AudioProcessing,
            _ => {}
        }
    }
    for cap in &model.capabilities {
        match cap.as_str() {
            "speech-to-text" => return AudioTask::SpeechToText,
            "text-to-speech" => return AudioTask::TextToSpeech,
            "audio-generation" => return AudioTask::SoundGeneration,
            _ => {}
        }
    }
    if model.has_file("preprocessor_config.json") && model.has_file("tokenizer.json") {
        return AudioTask::SpeechToText;
    }
    AudioTask::Unknown
}

pub struct AudioAdapter {
    base: AdapterBase,
}

impl AudioAdapter {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            base: AdapterBase::new(data_dir),
        }
    }
}

impl RuntimeAdapter for AudioAdapter {
    fn id(&self) -> &'static str {
        ID
    }

    fn label(&self) -> &'static str {
        "Audio (Whisper STT / Kokoro TTS)"
    }

    fn detect(&self, model: &Model) -> bool {
        classify_task(model) != AudioTask::Unknown
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
                format!("runtime bundle missing (v0.3 runtime): {}", script.display()),
                "Audio support lands in v0.3. Until then, run Whisper/Kokoro through \
                 the Transformers adapter from the model page.",
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
                format!("audio env install failed: {e}"),
                "Check disk space and network, then retry from Environments.",
            )
        })?;
        Ok(())
    }

    fn prepare(&self, model: &Model) -> RuntimeResult<()> {
        let _env = prepare_common(self.base.data_dir(), ID, model)?;
        if classify_task(model) == AudioTask::Unknown {
            return Err(RuntimeError::unsupported(
                "audio adapter needs a Whisper/Kokoro-style STT or TTS model",
            ));
        }
        Ok(())
    }

    fn run(&self, req: InferenceRequest) -> RuntimeResult<InferenceResult> {
        let env = prepare_common(self.base.data_dir(), ID, &req.model)?;
        let entry = require_serve_entrypoint(self.base.data_dir(), ID, &env)?;
        let task = classify_task(&req.model);
        let mut job_req = req;
        job_req
            .params
            .entry("audio_task".to_string())
            .or_insert_with(|| format!("{task:?}"));
        let job = job_json(ID, &job_req);
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
        // Convention: serve.py prints transcript text, or the output audio
        // path on its last line for TTS.
        let last = stdout.lines().last().unwrap_or("").trim().to_string();
        let last_path = PathBuf::from(&last);
        let (text, files) = if last_path.is_file()
            && last_path
                .extension()
                .map(|e| matches!(e.to_str(), Some("wav" | "mp3" | "ogg" | "flac")))
                .unwrap_or(false)
        {
            (None, vec![last_path])
        } else if last.is_empty() {
            (None, Vec::new())
        } else {
            (Some(stdout), Vec::new())
        };
        let mut sidecar = std::collections::HashMap::new();
        sidecar.insert("runtime".to_string(), ID.to_string());
        sidecar.insert("model".to_string(), job_req.model.repository.clone());
        Ok(InferenceResult {
            text,
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
