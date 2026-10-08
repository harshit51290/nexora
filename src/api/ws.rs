//! WebSocket streaming (`docs/10-API-CLI.md` §10.2, M14 slice).
//!
//! `GET /ws/generate?model=...&prompt=...` upgrades and streams the run:
//! `started` → `token` frames → `done`. Adapters are one-shot `run()` calls,
//! so tokens are word-chunks of the completed text (true token streaming
//! awaits a streaming adapter API — NEED); media runs emit `done` with the
//! output path and no token frames. Failures arrive as
//! `error {code,message,fix}` and then the socket closes.

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

use super::{core_stub, AppState, GenerateRequest};

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
    let (model, prompt) = match (params.model, params.prompt) {
        (Some(m), Some(p)) if !m.trim().is_empty() && !p.trim().is_empty() => (m, p),
        _ => {
            send(
                &mut socket,
                &serde_json::json!({
                    "type": "error",
                    "code": "E_BAD_WS_PARAMS",
                    "message": "need ?model=<id>&prompt=<text>, both non-empty",
                    "fix": "Reconnect with both query params, e.g. /ws/generate?model=m&prompt=hi.",
                    "version": version,
                }),
            )
            .await;
            return;
        }
    };
    send(
        &mut socket,
        &serde_json::json!({"type": "started", "model": model, "version": version}),
    )
    .await;

    let req = GenerateRequest {
        model: model.clone(),
        prompt,
        width: None,
        height: None,
        seed: None,
        params: None,
    };
    let result = match core_stub::generate(&req).await {
        Ok(r) => r,
        Err(e) => {
            send(
                &mut socket,
                &serde_json::json!({
                    "type": "error",
                    "code": e.code,
                    "message": e.message,
                    "fix": e.fix,
                    "version": version,
                }),
            )
            .await;
            return;
        }
    };

    // Token frames (M14 slice: word-chunks of the completed text).
    if let Some(text) = result.text.clone() {
        for (i, chunk) in text.split_whitespace().collect::<Vec<_>>().chunks(6).enumerate() {
            if send(
                &mut socket,
                &serde_json::json!({"type": "token", "index": i, "text": chunk.join(" ")}),
            )
            .await
            .is_err()
            {
                return;
            }
        }
    }
    send(
        &mut socket,
        &serde_json::json!({
            "type": "done",
            "id": result.id,
            "output_path": result.output_path,
            "runtime": result.sidecar.runtime,
            "version": version,
        }),
    )
    .await;
}

/// Send one JSON frame; `Err` when the client went away.
async fn send(socket: &mut WebSocket, frame: &serde_json::Value) -> Result<(), ()> {
    socket
        .send(Message::Text(frame.to_string()))
        .await
        .map_err(|_| ())
}
