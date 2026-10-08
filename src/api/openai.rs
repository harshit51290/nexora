//! OpenAI-compatible chat endpoint (`docs/10-API-CLI.md` §10.2).
//!
//! `POST /v1/chat/completions` on `http://localhost:8000` for text models so
//! existing apps point at the local runner unchanged. Messages are flattened
//! to a single prompt (real logic, kept after wiring); `temperature` and
//! `max_tokens` are forwarded as runtime params; the model row must carry a
//! text capability or the request is rejected with `E_CAPABILITY_MISMATCH`.
//! `stream: true` stays on the M14 WS path (`super::ws`) and is rejected
//! with `E_STREAMING_DEFERRED` here.

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
        return Err(ApiError::unimplemented(
            state.version,
            "E_STREAMING_DEFERRED",
            "stream:true is deferred to M14 WS streaming.".to_string(),
            "Retry with stream:false, or open /ws/generate for token frames.",
        ));
    }
    // Route to a TEXT runtime: chat on an image/audio model is a 400, not a
    // confusing runtime failure. Empty capability lists (legacy rows) pass
    // through and let adapter `detect()` decide.
    let record = core_stub::model_record(&req.model)
        .await
        .map_err(|e| ApiError::from_core(state.version, e))?;
    if !record.capabilities.is_empty()
        && !record
            .capabilities
            .iter()
            .any(|c| c == "text-generation" || c == "chat")
    {
        return Err(ApiError::bad_request(
            state.version,
            "E_CAPABILITY_MISMATCH",
            format!(
                "model {} has capabilities {:?}; chat needs text-generation.",
                req.model, record.capabilities
            ),
            "Point the request at a text/chat model (see GET /models).",
        ));
    }
    let prompt = messages_to_prompt(&req.messages);
    let prompt_tokens = estimate_tokens(&prompt);
    let gen_req = GenerateRequest {
        model: req.model.clone(),
        prompt,
        width: None,
        height: None,
        seed: None,
        execution: None,
        // temperature/max_tokens forwarded as runtime params; adapters
        // parse what they support and ignore the rest.
        params: Some(serde_json::json!({
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        })),
    };
    // Text runtime via scheduler + RuntimeAdapter prepare->run; usage counts
    // are tokenizer-free estimates (see `estimate_tokens`), not exact counts.
    let result = core_stub::generate(&gen_req)
        .await
        .map_err(|e| ApiError::from_core(state.version, e))?;
    let text = result.text.unwrap_or_default();
    let completion_tokens = estimate_tokens(&text);
    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: ChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// Tokenizer-free usage approximation (field names stay OpenAI-compatible:
/// `prompt_tokens` / `completion_tokens` / `total_tokens`).
///
/// ESTIMATED — adapters report no real token counts yet, so this counts
/// whitespace-separated words + ASCII punctuation marks + CJK characters
/// (each ~1 token in most BPE tokenizers). Punctuation matters: code and
/// prose heavy in `{};,:"` tokenize far above the old 4-chars-per-token
/// heuristic. Replace with real runtime counters once adapters report them.
fn estimate_tokens(s: &str) -> u32 {
    fn is_cjk(c: char) -> bool {
        matches!(
            c,
            '\u{4E00}'..='\u{9FFF}' | '\u{3040}'..='\u{30FF}' | '\u{AC00}'..='\u{D7AF}'
        )
    }
    let t = s.trim();
    if t.is_empty() {
        return 0;
    }
    // CJK scripts carry ~1 token per character with no whitespace splitting,
    // so count them exclusively: strip them before word/punct counting to
    // avoid double-counting a token as both a "word" and its characters.
    let cjk = t.chars().filter(|c| is_cjk(*c)).count() as u32;
    let rest: String = t.chars().filter(|c| !is_cjk(*c)).collect();
    let words = rest.split_whitespace().count() as u32;
    let punct = rest.chars().filter(|c| c.is_ascii_punctuation()).count() as u32;
    (words + punct + cjk).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero_nonempty_is_at_least_one() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("   "), 0);
        assert!(estimate_tokens("hi") >= 1);
    }

    #[test]
    fn punct_heavy_text_counts_above_word_count() {
        let words_only = estimate_tokens("hello world foo bar");
        let punct_heavy = estimate_tokens("hello, world! {foo: [bar];}");
        assert!(punct_heavy > words_only);
    }

    #[test]
    fn cjk_chars_count_per_char() {
        assert_eq!(estimate_tokens("你好"), 2);
    }
}
