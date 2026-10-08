//! Nexora binary — thin CLI entry over the orchestration library.

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
