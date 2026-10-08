//! WebSocket streaming stub (`docs/10-API-CLI.md` §10.2, M14).
//!
//! `GET /ws/generate?model=...&prompt=...` upgrades and currently sends a
//! single `hello` frame carrying `E_CORE_NOT_WIRED`, then drains inbound
//! until the client closes.
//!
//! TODO-CORE-WIRE (M14): replace the hello+close with the scheduler/job
//! progress feed — `Waiting/Running/progress/Completed` frames per
//! docs/10 §10.4 — and stream tokens/chunks for `/v1/chat/completions`.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::Deserialize;

use super::{core_stub::NOT_WIRED_FIX, AppState};

#[derive(Debug, Deserialize)]
pub struct WsParams {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
}

/// Routes mounted at `/` (see [`super::build_router`]).
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/ws/generate", get(ws_handler))
        .with_state(state)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<WsParams>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let version = state.version;
    ws.on_upgrade(move |socket| handle_socket(socket, version, params))
}

async fn handle_socket(mut socket: WebSocket, version: &'static str, params: WsParams) {
    let prompt_len = params.prompt.as_ref().map(|p| p.len()).unwrap_or(0);
    let hello = serde_json::json!({
        "type": "hello",
        "code": "E_CORE_NOT_WIRED",
        "message": format!(
            "TODO-CORE-WIRE: streaming not wired yet (model={:?}, prompt_len={}).",
            params.model, prompt_len
        ),
        "fix": NOT_WIRED_FIX,
        "version": version,
    });
    if socket.send(Message::Text(hello.to_string())).await.is_err() {
        return;
    }
    // Drain inbound until the client goes away; keeps the socket tidy.
    while let Some(Ok(_)) = socket.recv().await {}
    let _ = socket.send(Message::Close(None)).await;
}
