//! `uar` binary entry (`docs/10-API-CLI.md` §10.3).
//!
//! Thin wrapper: argument parsing + dispatch live in [`nexora::cli`], which
//! calls the same core fns as the API/Tauri layers (all TODO-CORE-WIRE
//! stubs until the core managers land).

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    nexora::cli::run().await
}
