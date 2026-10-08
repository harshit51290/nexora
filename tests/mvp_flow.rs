//! Victory-1 slice (`docs/03-ROADMAP-MILESTONES.md` §3.1):
//! `HF URL -> Analyze -> Install -> Configure -> Run -> Generate`.
//!
//! Core managers are stubs, so this file pins the *contracts* that survive
//! wiring: HF-URL parsing, analyzer routing (SD1.5 -> Diffusers,
//! Qwen-GGUF -> llama.cpp, unknown -> unsupported card), stub error codes,
//! the output-sidecar shape (docs/04 §4.5), OpenAI message mapping, the job
//! queue lifecycle, and workflow-template validity.

use nexora::api::{
    core_stub, openai::ChatMessage, openai::message_text, openai::messages_to_prompt,
    GenerationSidecar,
};
use nexora::jobs::{BatchSpec, JobKind, JobQueue, JobStatus};
use serde_json::json;

// ---------------------------------------------------------------------------
// HF URL -> repo id (victory-1 step 1: "paste SD URL")
// ---------------------------------------------------------------------------

#[test]
fn hf_url_parses_full_sd_url() {
    assert_eq!(
        core_stub::parse_hf_url("https://huggingface.co/runwayml/stable-diffusion-v1-5").unwrap(),
        "runwayml/stable-diffusion-v1-5"
    );
}

#[test]
fn hf_url_tolerates_git_suffix_and_deep_links() {
    assert_eq!(
        core_stub::parse_hf_url("https://huggingface.co/runwayml/stable-diffusion-v1-5.git")
            .unwrap(),
        "runwayml/stable-diffusion-v1-5"
    );
    assert_eq!(
        core_stub::parse_hf_url(
            "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/README.md"
        )
        .unwrap(),
        "runwayml/stable-diffusion-v1-5"
    );
    assert_eq!(
        core_stub::parse_hf_url("https://hf.co/Qwen/Qwen2-7B-Instruct-GGUF").unwrap(),
        "Qwen/Qwen2-7B-Instruct-GGUF"
    );
}

#[test]
fn hf_url_accepts_bare_repo_id() {
    assert_eq!(
        core_stub::parse_hf_url("runwayml/stable-diffusion-v1-5").unwrap(),
        "runwayml/stable-diffusion-v1-5"
    );
}

#[test]
fn hf_url_rejects_garbage_with_machine_code() {
    for bad in ["", "not-a-url", "https://example.com/foo/bar", "justoneword", "a/b/c"] {
        let err = core_stub::parse_hf_url(bad).unwrap_err();
        assert_eq!(err.code, "E_BAD_HF_URL", "input: {bad:?}");
        assert!(!err.fix.is_empty(), "every error carries a human fix");
    }
}

// ---------------------------------------------------------------------------
// Analyze (victory-1 step 2: "Stable Diffusion/Diffusers/4GB low-VRAM cfg")
// ---------------------------------------------------------------------------

#[test]
fn analyzer_routes_sd_to_diffusers() {
    let rec = core_stub::classify_model("runwayml/stable-diffusion-v1-5");
    assert_eq!(rec.runtime, "diffusers");
    assert_eq!(rec.task, "text-to-image");
}

#[test]
fn analyzer_routes_gguf_to_llama_cpp() {
    let rec = core_stub::classify_model("Qwen/Qwen2-7B-Instruct-GGUF");
    assert_eq!(rec.runtime, "llama.cpp");
    assert_eq!(rec.task, "text-generation");
}

#[test]
fn analyzer_routes_unknown_to_unsupported_card() {
    let rec = core_stub::classify_model("someuser/mystery-xyz-999");
    assert_eq!(rec.runtime, "unsupported");
    assert!(!rec.notes.is_empty());
}

// ---------------------------------------------------------------------------
// Service wiring (victory-1 steps 3-4: API/CLI -> real managers, Phase C).
// Hermetic: NEXORA_DATA_DIR points at a fresh temp dir, so no test touches
// the real data root and no network is needed (list/hardware/missing-row).
// ---------------------------------------------------------------------------

fn isolated_data_dir() -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("nexora-test-{}-{nanos}", std::process::id()))
}

#[tokio::test]
async fn install_load_generate_are_wired_to_service() {
    let dir = isolated_data_dir();
    std::env::set_var("NEXORA_DATA_DIR", &dir);

    // Fresh DB bootstrap + query through ModelManager (no models yet).
    let models = core_stub::list_models()
        .await
        .expect("list_models wires to the service layer");
    assert!(models.is_empty());

    // Local-only detection through HardwareBackend (no network).
    let _hw = core_stub::hardware_info()
        .await
        .expect("hardware_info wires to the service layer");

    // Unknown id -> coded manager error, never E_CORE_NOT_WIRED.
    let err = core_stub::load_model("no-such-model").await.unwrap_err();
    assert_ne!(err.code, "E_CORE_NOT_WIRED");
    assert!(!err.message.is_empty(), "coded error needs a message");
    assert!(!err.fix.is_empty(), "coded error needs a human fix");

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Output sidecar shape (docs/04 §4.5 reproducibility contract)
// ---------------------------------------------------------------------------

#[test]
fn sidecar_carries_reproducibility_fields() {
    let sidecar = GenerationSidecar::new(
        "runwayml/stable-diffusion-v1-5".to_string(),
        "cyberpunk warrior".to_string(),
        Some(42),
        json!({"width": 512, "height": 512}),
        "diffusers".to_string(),
    );
    let v = serde_json::to_value(&sidecar).unwrap();
    for key in ["model", "prompt", "seed", "params", "timestamp", "runtime"] {
        assert!(v.get(key).is_some(), "sidecar missing {key}");
    }
    assert_eq!(v["seed"], json!(42));
    // Round-trips (history Reuse/Regenerate reads these back).
    let back: GenerationSidecar = serde_json::from_value(v).unwrap();
    assert_eq!(back.runtime, "diffusers");
}

// ---------------------------------------------------------------------------
// OpenAI message -> prompt mapping (docs/10 §10.2)
// ---------------------------------------------------------------------------

fn msg(role: &str, content: serde_json::Value) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content,
    }
}

#[test]
fn openai_messages_flatten_with_roles() {
    let prompt = messages_to_prompt(&[
        msg("system", json!("you are helpful")),
        msg("user", json!("draw a cat")),
    ]);
    assert!(prompt.contains("system: you are helpful"));
    assert!(prompt.contains("user: draw a cat"));
}

#[test]
fn openai_content_parts_join_text() {
    let content = json!([
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"}
    ]);
    assert_eq!(message_text(&content), "hello world");
    // Non-string scalars render instead of panicking.
    assert!(!message_text(&json!(7)).is_empty());
}

// ---------------------------------------------------------------------------
// Jobs lifecycle + batch (docs/10 §10.4)
// ---------------------------------------------------------------------------

#[test]
fn job_queue_lifecycle() {
    let mut q = JobQueue::new();
    let id = q.submit(JobKind::Image, None, "a cat".to_string()).id.clone();
    assert_eq!(q.get(&id).unwrap().status, JobStatus::Waiting);
    assert!(q.set_status(&id, JobStatus::Running));
    assert!(q.set_status(&id, JobStatus::Completed));
    assert_eq!(q.get(&id).unwrap().status, JobStatus::Completed);
    assert!(q.cancel(&id));
    assert_eq!(q.get(&id).unwrap().status, JobStatus::Cancelled);
    assert!(!q.cancel("job-999"));
    assert_eq!(q.list().len(), 1);
}

#[test]
fn batch_spec_validates_and_expands() {
    let bad = BatchSpec {
        name: "empty".to_string(),
        model: "m".to_string(),
        kind: JobKind::Image,
        prompts: vec![],
        outputs_per_prompt: 1,
    };
    assert!(bad.validate().is_err());
    let zero = BatchSpec {
        prompts: vec!["p".to_string()],
        outputs_per_prompt: 0,
        ..bad.clone()
    };
    assert!(zero.validate().is_err());

    let spec = BatchSpec {
        name: "b".to_string(),
        model: "runwayml/stable-diffusion-v1-5".to_string(),
        kind: JobKind::Image,
        prompts: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        outputs_per_prompt: 2,
    };
    assert!(spec.validate().is_ok());
    assert_eq!(spec.total_outputs(), 6);
    let jobs = spec.expand();
    assert_eq!(jobs.len(), 6);
    assert!(jobs.iter().all(|j| j.status == JobStatus::Waiting));

    let mut q = JobQueue::new();
    let ids = q.submit_batch(&spec).unwrap();
    assert_eq!(ids.len(), 6);
    assert_eq!(q.list().len(), 6);
}

// ---------------------------------------------------------------------------
// Workflow templates (docs/10 §10.5): 8 shipped node graphs stay valid
// ---------------------------------------------------------------------------

fn check_template(name: &str, raw: &str) {
    let v: serde_json::Value = serde_json::from_str(raw).unwrap();
    assert_eq!(v["name"], json!(name));
    assert_eq!(v["version"], json!("0.1.0"));
    let nodes = v["nodes"].as_array().unwrap();
    assert!(nodes.len() >= 2, "{name}: needs >= 2 nodes");
    let ids: std::collections::HashSet<&str> =
        nodes.iter().map(|n| n["id"].as_str().unwrap()).collect();
    assert_eq!(ids.len(), nodes.len(), "{name}: duplicate node id");
    let edges = v["edges"].as_array().unwrap();
    assert!(!edges.is_empty(), "{name}: needs >= 1 edge");
    for e in edges {
        assert!(ids.contains(e["from"].as_str().unwrap()), "{name}: dangling edge");
        assert!(ids.contains(e["to"].as_str().unwrap()), "{name}: dangling edge");
    }
    assert!(
        nodes.iter().any(|n| n["type"] == json!("save-output")),
        "{name}: needs a save-output sink"
    );
}

#[test]
fn workflow_templates_are_valid_graphs() {
    check_template(
        "text-to-image",
        include_str!("../workflows/templates/text-to-image.json"),
    );
    check_template(
        "image-to-image",
        include_str!("../workflows/templates/image-to-image.json"),
    );
    check_template(
        "text-to-speech",
        include_str!("../workflows/templates/text-to-speech.json"),
    );
    check_template(
        "speech-to-text",
        include_str!("../workflows/templates/speech-to-text.json"),
    );
    check_template(
        "text-to-video",
        include_str!("../workflows/templates/text-to-video.json"),
    );
    check_template(
        "image-to-video",
        include_str!("../workflows/templates/image-to-video.json"),
    );
    check_template(
        "llm-to-image",
        include_str!("../workflows/templates/llm-to-image.json"),
    );
    check_template(
        "llm-to-tts",
        include_str!("../workflows/templates/llm-to-tts.json"),
    );
}
