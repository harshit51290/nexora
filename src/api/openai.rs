//! OpenAI-compatible chat endpoint (`docs/10-API-CLI.md` §10.2).
//!
//! `POST /v1/chat/completions` on `http://localhost:8000` for text models so
//! existing apps point at the local runner unchanged. Messages are flattened
//! to a single prompt (real logic, kept after wiring); the runtime call is a
//! TODO-CORE-WIRE stub. `stream: true` is deferred to M14 WS streaming and
//! rejected with `E_STREAMING_DEFERRED` until then.

use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};

use super::{core_stub, ApiError, AppState, GenerateRequest};

// ---------------------------------------------------------------------------
// Wire types (subset of the OpenAI chat schema; unknown fields ignored)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// `string` or `[{type:"text", text:"..."}]` — both accepted.
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

// ---------------------------------------------------------------------------
// Message -> prompt mapping (REAL logic; survives core wiring)
// ---------------------------------------------------------------------------

/// Extract display text from a message `content` value (plain string,
/// `[{type:"text", text}]` parts array, or fallback debug rendering).
pub fn message_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .map(|p| match p {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(_) => p
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => content.to_string(),
    }
}

/// Flatten a chat transcript into the single prompt handed to the runtime.
/// `system` lines stay labeled so instruct models keep the separation.
pub fn messages_to_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, message_text(&m.content).trim()))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------

/// Routes mounted at `/` (see [`super::build_router`]).
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad_request(
            state.version,
            "E_EMPTY_MESSAGES",
            "Chat request has no messages.".to_string(),
            "Send at least one message, e.g. {\"role\":\"user\",\"content\":\"hi\"}.",
        ));
    }
    if req.stream {
        // TODO-CORE-WIRE (M14): token streaming over WS; see `super::ws`.
        return Err(ApiError::unimplemented(
            state.version,
            "E_STREAMING_DEFERRED",
            "stream:true is deferred to M14 WS streaming.".to_string(),
            "Retry with stream:false; non-streaming chat works once the text runtime is wired.",
        ));
    }
    let prompt = messages_to_prompt(&req.messages);
    let gen_req = GenerateRequest {
        model: req.model.clone(),
        prompt,
        width: None,
        height: None,
        seed: None,
        // TODO-CORE-WIRE: forward temperature/max_tokens as runtime params.
        params: Some(serde_json::json!({
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        })),
    };
    // TODO-CORE-WIRE: route to the text runtime (transformers/llama.cpp)
    // via RuntimeAdapter prepare->run; usage counters come from the runtime.
    let result = core_stub::generate(&gen_req)
        .await
        .map_err(|e| ApiError::from_stub(state.version, e))?;
    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: result.text.unwrap_or_default(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: ChatUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }))
}
