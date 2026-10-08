//! `uar` CLI (`docs/10-API-CLI.md` §10.3): ships with the desktop app.
//!
//! `uar install <owner/model> | uar run <...> | uar models | uar hardware |
//! uar generate --model ... --prompt "..." | uar serve`, plus `jobs`/`batch`
//! previews for §10.4. The CLI calls the SAME core fns as the API/Tauri
//! layers — today that means [`crate::api::core_stub`], so every command
//! currently exits via `E_CORE_NOT_WIRED` (TODO-CORE-WIRE) except local-only
//! work (arg parsing, prompts-file reading, sidecar writing), which is real.

use crate::api::{self, AppState, GenerateRequest};
use anyhow::Context;
use clap::{Parser, Subcommands};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "uar",
    version,
    about = "Universal AI Runner — install, run and generate with local models"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommands)]
pub enum Commands {
    /// Install a model from a Hugging Face repo id or URL.
    Install {
        /// Repo id (`owner/model`) or full URL (`https://huggingface.co/...`).
        repo: String,
        /// Model revision (branch/tag/commit); defaults to `main`.
        #[arg(long)]
        revision: Option<String>,
    },
    /// Load a model and run a one-shot prompt (loads runtime if needed).
    Run {
        /// Installed model id.
        model: String,
        /// Prompt; defaults to a hello-world probe once wired.
        #[arg(long)]
        prompt: Option<String>,
    },
    /// List installed models.
    Models,
    /// Show detected hardware (CPU/RAM/GPU/VRAM/CUDA/driver/storage).
    Hardware,
    /// Generate with an installed model (text or image per capability).
    Generate {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        width: Option<u32>,
        #[arg(long)]
        height: Option<u32>,
        #[arg(long)]
        seed: Option<u64>,
        /// Where to write the `<output>.json` sidecar (docs/04 §4.5);
        /// defaults to `<output_path>.json`.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Serve the local REST + OpenAI-compat API + WS endpoint.
    Serve {
        /// Bind address (default `127.0.0.1:8000`, docs/10 §10.2).
        #[arg(long, default_value = "127.0.0.1:8000")]
        bind: String,
    },
    /// List queued/batch jobs (stub queue until the scheduler lands).
    Jobs,
    /// Batch: N prompts (one per line) -> N outputs each (docs/10 §10.4).
    Batch {
        #[arg(long)]
        model: String,
        /// Text file with one prompt per line (e.g. 100 prompts).
        #[arg(long)]
        prompts: PathBuf,
        #[arg(long, default_value_t = 1)]
        outputs_per_prompt: u32,
    },
}

/// CLI entry point (called by the `uar` binary in `main.rs`).
pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Install { repo, revision } => {
            // TODO-CORE-WIRE: model_manager::install + download_manager.
            let id = api::core_stub::parse_hf_url(&repo).map_err(|e| anyhow::anyhow!("{e}"))?;
            let _ = api::core_stub::install_model(&id, revision.as_deref())
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("installed {id}");
            Ok(())
        }
        Commands::Run { model, prompt } => {
            // TODO-CORE-WIRE: model_manager::load then one-shot generate.
            let _ = api::core_stub::load_model(&model)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let req = GenerateRequest {
                model,
                prompt: prompt.unwrap_or_else(|| "hello world".to_string()),
                width: None,
                height: None,
                seed: None,
                params: None,
            };
            let _ = api::core_stub::generate(&req)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(())
        }
        Commands::Models => {
            // TODO-CORE-WIRE: SELECT FROM models (docs/04 §4.1).
            let _ = api::core_stub::list_models()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(())
        }
        Commands::Hardware => {
            // TODO-CORE-WIRE: HardwareBackend::detect (docs/07).
            let _ = api::core_stub::hardware_info()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(())
        }
        Commands::Generate {
            model,
            prompt,
            width,
            height,
            seed,
            out,
        } => {
            // TODO-CORE-WIRE: scheduler -> RuntimeAdapter prepare->run.
            let req = GenerateRequest {
                model,
                prompt,
                width,
                height,
                seed,
                params: None,
            };
            let result = api::core_stub::generate(&req)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // Real logic (runs post-wire): persist the output sidecar
            // (docs/04 §4.5) next to the artifact.
            let sidecar = serde_json::to_string_pretty(&result.sidecar)?;
            let dest = out.unwrap_or_else(|| PathBuf::from(format!("{}.json", result.output_path)));
            if let Some(parent) = dest.parent() {
                if !parent.as_os_str().is_empty() {
                    tokio::fs::create_dir_all(parent).await.with_context(|| {
                        format!("E_SIDECAR_MKDIR: cannot create {}", parent.display())
                    })?;
                }
            }
            tokio::fs::write(&dest, sidecar)
                .await
                .with_context(|| format!("E_SIDECAR_WRITE: cannot write {}", dest.display()))?;
            println!(
                "saved {} + sidecar {}",
                result.output_path,
                dest.display()
            );
            Ok(())
        }
        Commands::Serve { bind } => {
            let app = api::build_router(AppState::new());
            let listener = tokio::net::TcpListener::bind(&bind)
                .await
                .with_context(|| format!("E_BIND_FAILED: cannot listen on {bind}"))?;
            tracing::info!("uar serve on http://{bind}");
            println!("Serving API on http://{bind} (REST + /v1/* OpenAI-compat + /ws/generate)");
            // TODO-CORE-WIRE: none here — serving works today; only the
            // handlers behind it are stubs.
            axum::serve(listener, app)
                .await
                .context("E_SERVE_FAILED: axum server exited with an error")?;
            Ok(())
        }
        Commands::Jobs => {
            // NOTE: in-memory stub queue only. Do NOT grow this into a real
            // scheduler — VRAM-gated concurrency lives in `src/scheduler`
            // (docs/10 §10.4); see `crate::jobs` NOTE. TODO-CORE-WIRE.
            let queue = crate::jobs::JobQueue::new();
            println!(
                "jobs: {} queued (stub queue; scheduler not wired yet)",
                queue.list().len()
            );
            Ok(())
        }
        Commands::Batch {
            model,
            prompts,
            outputs_per_prompt,
        } => {
            // Real logic: read + validate the batch spec locally.
            let text = std::fs::read_to_string(&prompts).with_context(|| {
                format!(
                    "E_PROMPTS_UNREADABLE: cannot read prompts file {}",
                    prompts.display()
                )
            })?;
            let lines: Vec<String> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();
            let spec = crate::jobs::BatchSpec {
                name: format!("batch-{}-prompts", lines.len()),
                model,
                kind: crate::jobs::JobKind::Image,
                prompts: lines,
                outputs_per_prompt,
            };
            spec.validate()
                .map_err(|e| anyhow::anyhow!("[E_BATCH_INVALID] {e}"))?;
            // TODO-CORE-WIRE: hand BatchSpec to src/scheduler
            // (VRAM-gated concurrency, docs/10 §10.4). Nothing ran.
            anyhow::bail!(
                "[E_CORE_NOT_WIRED] TODO-CORE-WIRE: batch execution needs src/scheduler \
                 ({} prompts x{} = {} outputs staged, nothing ran)",
                spec.prompts.len(),
                spec.outputs_per_prompt,
                spec.total_outputs()
            );
        }
    }
}
