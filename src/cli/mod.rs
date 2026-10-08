//! `uar` CLI (`docs/10-API-CLI.md` §10.3): ships with the desktop app.
//!
//! `uar install <owner/model> | uar run <...> | uar models | uar hardware |
//! uar generate --model ... --prompt "..." | uar serve`, plus `jobs`/`batch`
//! for §10.4 and `downloads` for pause/resume/cancel. The CLI calls the SAME
//! core fns as the API/Tauri layers ([`crate::api::core_stub`]) — now real
//! manager calls (install returns while bytes move in the background).

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
    /// List queued/batch jobs (local queue view; live VRAM-gated
    /// admission lives in `src/scheduler`, docs/10 §10.4).
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
    /// Pause, resume or cancel the active download.
    Downloads {
        #[command(subcommand)]
        action: DownloadAction,
    },
}

#[derive(Debug, Clone, Subcommands)]
pub enum DownloadAction {
    /// Pause the active download (resume keeps `.part` offsets).
    Pause,
    /// Resume a paused download.
    Resume,
    /// Cancel the active download (state stays resumable).
    Cancel,
}

/// CLI entry point (called by the `uar` binary in `main.rs`).
pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Install { repo, revision } => {
            let id = api::core_stub::parse_hf_url(&repo).map_err(|e| anyhow::anyhow!("{e}"))?;
            let model_id = api::core_stub::install_model(&id, revision.as_deref())
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // Row is DOWNLOADING; bytes move in a background task.
            println!("installing {model_id} (DOWNLOADING in background)");
            Ok(())
        }
        Commands::Run { model, prompt } => {
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
            let result = api::core_stub::generate(&req)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            match result.text {
                Some(text) => println!("{text}"),
                None => println!("saved {}", result.output_path),
            }
            Ok(())
        }
        Commands::Models => {
            let models = api::core_stub::list_models()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            if models.is_empty() {
                println!("no models installed (uar install <owner/model>)");
            }
            for m in models {
                println!(
                    "{}\t{}\t{}\t[{}]",
                    m.id,
                    m.status,
                    m.repository,
                    m.capabilities.join(",")
                );
            }
            Ok(())
        }
        Commands::Hardware => {
            let hw = api::core_stub::hardware_info()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("{}", hw.label);
            println!(
                "vram: {} / ram: {}",
                hw.gpu_vram_gb
                    .map(|g| format!("{g:.1}GB"))
                    .unwrap_or_else(|| "n/a (CPU)".into()),
                hw.system_ram_gb
                    .map(|g| format!("{g:.1}GB"))
                    .unwrap_or_else(|| "n/a".into()),
            );
            println!("notes: {}", hw.notes);
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
            // Scheduler -> RuntimeAdapter prepare->run (same core fn as
            // POST /generate). The core already writes the output sidecar
            // (docs/04 §4.5); `--out` copies it to the requested path.
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
            if let Some(text) = &result.text {
                println!("{text}");
            }
            // Empty-run edge: the artifact IS the sidecar file already.
            let sidecar_src = if result.output_path.ends_with(".json") {
                PathBuf::from(&result.output_path)
            } else {
                PathBuf::from(format!("{}.json", result.output_path))
            };
            if let Some(dest) = out {
                if sidecar_src != dest {
                    if let Some(parent) = dest.parent() {
                        if !parent.as_os_str().is_empty() {
                            tokio::fs::create_dir_all(parent).await.with_context(|| {
                                format!("E_SIDECAR_MKDIR: cannot create {}", parent.display())
                            })?;
                        }
                    }
                    tokio::fs::copy(&sidecar_src, &dest)
                        .await
                        .with_context(|| {
                            format!(
                                "E_SIDECAR_WRITE: cannot copy {} to {}",
                                sidecar_src.display(),
                                dest.display()
                            )
                        })?;
                    println!(
                        "saved {} + sidecar {}",
                        result.output_path,
                        dest.display()
                    );
                } else {
                    println!(
                        "saved {} + sidecar {}",
                        result.output_path,
                        sidecar_src.display()
                    );
                }
            } else {
                println!(
                    "saved {} + sidecar {}",
                    result.output_path,
                    sidecar_src.display()
                );
            }
            Ok(())
        }
        Commands::Serve { bind } => {
            let app = api::build_router(AppState::new());
            let listener = tokio::net::TcpListener::bind(&bind)
                .await
                .with_context(|| format!("E_BIND_FAILED: cannot listen on {bind}"))?;
            tracing::info!("uar serve on http://{bind}");
            println!("Serving API on http://{bind} (REST + /v1/* OpenAI-compat + /ws/generate)");
            // Handlers are real manager calls; the DB boots on first request.
            axum::serve(listener, app)
                .await
                .context("E_SERVE_FAILED: axum server exited with an error")?;
            Ok(())
        }
        Commands::Jobs => {
            // NOTE: in-memory queue view only. Do NOT grow this into a real
            // scheduler — VRAM-gated concurrency lives in `src/scheduler`
            // (docs/10 §10.4); see `crate::jobs` NOTE.
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
            // Sequential runs through the same core fn as POST /generate
            // (scheduler admission applies per item; VRAM-gated).
            let mut ok = 0usize;
            let mut failed = 0usize;
            for (i, prompt) in spec.prompts.iter().enumerate() {
                for n in 0..spec.outputs_per_prompt {
                    let req = GenerateRequest {
                        model: spec.model.clone(),
                        prompt: prompt.clone(),
                        width: None,
                        height: None,
                        seed: None,
                        params: None,
                    };
                    match api::core_stub::generate(&req).await {
                        Ok(result) => {
                            ok += 1;
                            println!("batch item {}/{n}: saved {}", i + 1, result.output_path);
                        }
                        Err(e) => {
                            failed += 1;
                            eprintln!("batch item {}/{n} failed: {e}", i + 1);
                        }
                    }
                }
            }
            println!("batch done: {ok} ok, {failed} failed");
            if failed > 0 {
                anyhow::bail!("[E_BATCH_PARTIAL] {failed} batch items failed (see lines above)");
            }
            Ok(())
        }
        Commands::Downloads { action } => {
            let name = match action {
                DownloadAction::Pause => "pause",
                DownloadAction::Resume => "resume",
                DownloadAction::Cancel => "cancel",
            };
            let state = api::core_stub::download_control(name)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("downloads: {state}");
            Ok(())
        }
    }
}
