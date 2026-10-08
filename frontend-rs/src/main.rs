//! nexora-ui — pure-Rust desktop UI for Nexora (Dioxus track).
//!
//! Same `nexora` core as the REST/CLI layers, called directly (no IPC hop).
//! This IS the desktop UI (the legacy React `frontend/` was removed).

mod app;
mod backend;
mod components;
mod pages;
mod state;

fn main() {
    dioxus_desktop::launch(app::App);
}
