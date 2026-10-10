//! `uar` CLI (`docs/10-API-CLI.md` §10.3): ships with the desktop app.
//!
//! `uar install <owner/model> | uar run <...> | uar models | uar hardware |
//! uar generate --model ... --prompt "..." | uar serve`, plus `jobs`/`batch`
//! for §10.4 and `downloads` for pause/resume/cancel. The CLI calls the SAME
//! core fns as the API/Tauri layers ([`crate::api::core_stub`]) — now real
//! manager calls (install returns while bytes move in the background).

use crate::api::{self, AppState, ExecutionPrefs, GenerateRequest};
use anyhow::Context;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "uar",
    version,
    about = "Universal AI Runner — install, run and generate with local models"
)]
pub struct Cli {
    /// Subcommand. When omitted the CLI drops into an interactive REPL
    /// instead of exiting (stays open when double-clicked on Windows).
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
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
        /// Force CPU execution: skip the VRAM gate and run fully on CPU.
        /// Trade-off: much slower inference and high RAM use; use when the
        /// model cannot fit VRAM even with offload.
        #[arg(long)]
        cpu: bool,
        /// Widen the VRAM gate: proceed with CPU offload even past the normal
        /// offload window instead of hard-failing with E_VRAM_SHORT.
        /// Trade-off: slow tokens and RAM pressure that can still OOM on
        /// small machines.
        #[arg(long)]
        offload: bool,
        /// Layers to keep on GPU when offloading (llama.cpp style, e.g. 20);
        /// the rest run on CPU. Trade-off: more layers = faster but more
        /// VRAM; 0 = full CPU.
        #[arg(long)]
        gpu_layers: Option<u32>,
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
        /// Force CPU execution: skip the VRAM gate and run fully on CPU.
        /// Trade-off: much slower inference and high RAM use; use when the
        /// model cannot fit VRAM even with offload.
        #[arg(long)]
        cpu: bool,
        /// Widen the VRAM gate: proceed with CPU offload even past the normal
        /// offload window instead of hard-failing with E_VRAM_SHORT.
        /// Trade-off: slow tokens and RAM pressure that can still OOM on
        /// small machines.
        #[arg(long)]
        offload: bool,
        /// Layers to keep on GPU when offloading (llama.cpp style, e.g. 20);
        /// the rest run on CPU. Trade-off: more layers = faster but more
        /// VRAM; 0 = full CPU.
        #[arg(long)]
        gpu_layers: Option<u32>,
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
    /// Estimate inference memory for a Hub repo WITHOUT downloading it
    /// (reads Safetensors headers / GGUF metadata via HTTP Range requests;
    /// hf-mem method, see `src/mem`). Powers the pre-download estimate card.
    Estimate {
        /// Repo id (`owner/model`) or full URL (`https://huggingface.co/...`).
        repo: String,
        /// Model revision (branch/tag/commit); defaults to `main`.
        #[arg(long)]
        revision: Option<String>,
        /// Also estimate KV-cache bytes for CausalLM/ConditionalGeneration
        /// (needs `config.json` attention dims + context length).
        #[arg(long)]
        experimental: bool,
        /// Context length override for the KV estimate (else config default).
        #[arg(long)]
        max_model_len: Option<u64>,
        /// Batch size for the KV estimate (default 1).
        #[arg(long, default_value_t = 1)]
        batch_size: u64,
        /// KV-cache dtype (`auto` + choices in `src/mem`; GGUF: F16/Q4_K…).
        #[arg(long, default_value = "auto")]
        kv_cache_dtype: String,
        /// Single GGUF file to estimate (else every GGUF file is listed).
        #[arg(long)]
        gguf_file: Option<String>,
        /// Print the full estimate as JSON (all bytes, dtypes, MoE, KV).
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum DownloadAction {
    /// Pause the active download (resume keeps `.part` offsets).
    Pause,
    /// Resume a paused download.
    Resume,
    /// Cancel the active download (state stays resumable).
    Cancel,
}

/// CLI entry point (called by the `uar` binary in `main.rs`).
///
/// With a subcommand this dispatches once and exits; with no arguments it
/// enters the interactive REPL (stays open when launched without args).
pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(cmd) => dispatch(cmd).await,
        None => repl().await,
    }
}

/// Interactive REPL: read `uar <args>` lines until `quit`/`exit`/EOF.
/// Each line is parsed with the same Clap definition as argv, so every
/// subcommand works unchanged (`hardware`, `estimate gpt2 --json`, …).
/// Commands that never return (`serve`) run until Ctrl-C, then the REPL
/// continues.
pub async fn repl() -> anyhow::Result<()> {
    use std::io::{BufRead, Write};
    println!("uar interactive mode — type `help` for commands, `quit` to exit.");
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("uar> ");
        std::io::stdout().flush().ok();
        let line = match lines.next() {
            Some(Ok(line)) => line,
            _ => break, // EOF / stdin closed
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" {
            break;
        }
        let mut argv = vec!["uar".to_string()];
        argv.extend(split_repl_line(line));
        match Cli::try_parse_from(argv) {
            Ok(cli) => match cli.command {
                Some(cmd) => {
                    if let Err(e) = dispatch(cmd).await {
                        eprintln!("error: {e:#}");
                    }
                }
                None => continue,
            },
            Err(e) => {
                // Clap renders help / parse errors itself.
                let _ = e.print();
            }
        }
    }
    println!("bye.");
    Ok(())
}

/// Minimal quote-aware splitter for REPL lines: whitespace separates args
/// unless inside double quotes (`generate --model m --prompt "hello world"`
/// keeps the prompt as one arg). No escape processing — keep it predictable.
fn split_repl_line(line: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
            }
            c if c.is_whitespace() && !in_quotes => {
                if !cur.is_empty() {
                    args.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        args.push(cur);
    }
    args
}

async fn dispatch(cmd: Commands) -> anyhow::Result<()> {
    match cmd {
        Commands::Install { repo, revision } => {
            let id = api::core_stub::parse_hf_url(&repo).map_err(|e| anyhow::anyhow!("{e}"))?;
            let model_id = api::core_stub::install_model(&id, revision.as_deref())
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // Row is DOWNLOADING; bytes move in a background task.
            println!("installing {model_id} (DOWNLOADING in background)");
            Ok(())
        }
        Commands::Run {
            model,
            prompt,
            cpu,
            offload,
            gpu_layers,
        } => {
            let prefs = ExecutionPrefs {
                cpu,
                offload,
                gpu_layers,
            };
            api::core_stub::load_model_with(&model, &prefs)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let req = GenerateRequest {
                model,
                prompt: prompt.unwrap_or_else(|| "hello world".to_string()),
                width: None,
                height: None,
                seed: None,
                execution: Some(prefs),
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
            cpu,
            offload,
            gpu_layers,
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
                execution: Some(ExecutionPrefs {
                    cpu,
                    offload,
                    gpu_layers,
                }),
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
            // Persisted queue (DB truth; memory is only a cache) — the
            // scheduler owns admission, this only lists (docs/10 §10.4).
            let jobs = api::core_stub::list_jobs()
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("jobs: {} total (persisted)", jobs.len());
            println!("{:<10} {:<7} {:<10} {:<24} prompt", "id", "kind", "status", "model");
            for j in &jobs {
                let prompt: String = j.prompt.chars().take(48).collect();
                println!(
                    "{:<10} {:<7} {:<10} {:<24} {}",
                    j.id,
                    j.kind.as_str(),
                    j.status.db_str(),
                    j.model.as_deref().unwrap_or("-"),
                    prompt,
                );
            }
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
                        execution: None,
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
        Commands::Estimate {
            repo,
            revision,
            experimental,
            max_model_len,
            batch_size,
            kv_cache_dtype,
            gguf_file,
            json,
        } => {
            // Accept URL, `owner/model`, or a bare name (Hub alias,
            // e.g. `gpt2`) via the same shared helper as the API path —
            // one behavior, one error message, no drift.
            let mut parsed = match api::core_stub::parse_hf_url(&repo) {
                Ok(id) => {
                    let (owner, name) = id.split_once('/').unwrap_or(("", id.as_str()));
                    crate::hf::HfRepo {
                        owner: owner.to_string(),
                        repo: name.to_string(),
                        rev: "main".to_string(),
                    }
                }
                Err(_) if !repo.contains('/') && !repo.contains("://") => {
                    crate::hf::resolve_bare_name(&repo)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))?
                }
                Err(e) => return Err(anyhow::anyhow!("{e}")),
            };
            if let Some(rev) = revision {
                parsed.rev = rev;
            }
            if parsed.owner.is_empty() {
                // Fully-qualified ids only from here (Hub alias above
                // always returns `owner/model`).
                return Err(anyhow::anyhow!(
                    "[E_BAD_HF_URL] Pass a repo id (owner/model) or URL, e.g. https://huggingface.co/runwayml/stable-diffusion-v1-5."
                ));
            }
            let opts = crate::mem::EstimateOpts {
                experimental,
                max_model_len,
                batch_size,
                kv_cache_dtype,
                gguf_file,
            };
            let est = crate::mem::estimate_repo(&parsed, &opts)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&est).unwrap_or_default());
                return Ok(());
            }
            let gb = |b: u64| format!("{:.2} GB", b as f64 / 1024.0 / 1024.0 / 1024.0);
            println!("{} @ {}", est.model_id, est.revision);
            println!("weights: {} ({} params)", gb(est.weights_bytes), est.param_count);
            match (est.kv_bytes, est.total_bytes) {
                (Some(kv), Some(total)) => {
                    println!("kv-cache: {} ({})", gb(kv), est.kv_dtype.clone().unwrap_or_default());
                    println!("total:     {}", gb(total));
                }
                _ if experimental => println!(
                    "kv-cache: n/a (architecture has no CausalLM attention dims — see logs)"
                ),
                _ => println!("kv-cache: n/a (pass --experimental for CausalLM/VLM)"),
            }
            for f in &est.per_file {
                println!("  {:<48} {}", f.name, gb(f.bytes));
            }
            if !est.dtype_breakdown.is_empty() {
                println!("dtypes:");
                for row in &est.dtype_breakdown {
                    println!("  {:<10} {:>14} params  {}", row.dtype, row.params, gb(row.bytes));
                }
            }
            if let Some(moe) = &est.moe {
                println!(
                    "moe: base {} ({} params) + {} experts {} ({} params){}",
                    gb(moe.base_bytes),
                    moe.base_params,
                    moe.expert_count,
                    gb(moe.experts_total_bytes),
                    moe.experts_total_params,
                    moe.active_expert_count
                        .map(|n| format!(", {n} active"))
                        .unwrap_or_default()
                );
            }
            if let Some(sug) = &est.offload_suggestion {
                println!("offload: {sug}");
            }
            if experimental {
                println!("{}", serde_json::to_string_pretty(&est).unwrap_or_default());
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repl_splitter_handles_quotes_and_whitespace() {
        assert_eq!(split_repl_line("hardware"), vec!["hardware"]);
        assert_eq!(
            split_repl_line("generate --model m --prompt \"hello world\""),
            vec!["generate", "--model", "m", "--prompt", "hello world"]
        );
        assert!(split_repl_line("   ").is_empty());
    }

    #[test]
    fn repl_accepts_no_subcommand_without_exiting() {
        // `uar` with no args must enter the REPL, not fail to parse.
        let cli = Cli::try_parse_from(["uar"]).expect("bare uar parses");
        assert!(cli.command.is_none());
    }
}
