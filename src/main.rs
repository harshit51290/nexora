//! Nexora binary — thin CLI entry over the orchestration library.
//!
//! Runs on a dedicated thread with an 8MB stack (same Windows stack-overflow
//! fix as the `uar` entry: the default 1MB main-thread stack is too small
//! for large async Future state machines).

use clap::Parser;
use tracing_subscriber::EnvFilter;

/// Universal local AI runtime manager (orchestrator).
#[derive(Debug, Parser)]
#[command(name = "nexora", version)]
struct Cli {
    /// Data root (relocatable, e.g. `D:\AI\Models`). Defaults to OS data dir / Nexora.
    #[arg(long)]
    data_root: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let child = std::thread::Builder::new()
        .name("nexora-main".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .map_err(|e| anyhow::anyhow!("nexora: cannot spawn main thread: {e}"))?;
    child
        .join()
        .map_err(|_| anyhow::anyhow!("nexora: main thread panicked"))?
}

fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init()
        .ok();

    let cli = Cli::parse();
    let layout = match cli.data_root {
        Some(root) => nexora::storage::layout::StorageLayout::new(root),
        None => nexora::storage::layout::StorageLayout::default_root(),
    };
    tracing::info!(root = %layout.root.display(), "nexora orchestrator starting");
    Ok(())
}
