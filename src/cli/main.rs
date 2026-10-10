//! `uar` binary entry (`docs/10-API-CLI.md` §10.3).
//!
//! Thin wrapper: argument parsing + dispatch live in [`nexora::cli`], which
//! calls the same core fns as the API/Tauri layers (all TODO-CORE-WIRE
//! stubs until the core managers land).
//!
//! Windows stack fix: the default 1MB main-thread stack overflows on large
//! async Future state machines, so the Tokio runtime runs on a dedicated
//! thread with an 8MB stack.

fn main() -> anyhow::Result<()> {
    let child = std::thread::Builder::new()
        .name("uar-main".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            // 8MB on the main thread AND every worker: install/generate
            // futures overflow the 2MB defaults (a worker overflow aborts
            // the whole server process — seen live on install).
            let rt = tokio::runtime::Builder::new_multi_thread()
                .thread_stack_size(8 * 1024 * 1024)
                .enable_all()
                .build()
                .map_err(|e| anyhow::anyhow!("uar: cannot start async runtime: {e}"))?;
            rt.block_on(nexora::cli::run())
        })
        .map_err(|e| anyhow::anyhow!("uar: cannot spawn main thread: {e}"))?;
    child
        .join()
        .map_err(|_| anyhow::anyhow!("uar: main thread panicked"))?
}
