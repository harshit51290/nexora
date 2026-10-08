//! nexora-ui — pure-Rust desktop UI for Nexora (Dioxus track).
//!
//! Same `nexora` core as the REST/CLI layers, called directly (no IPC hop).
//! React frontend in `frontend/` stays until this track reaches parity.

mod app;
mod backend;
mod components;
mod pages;
mod state;

fn main() {
    dioxus_desktop::launch(app::App);
}
