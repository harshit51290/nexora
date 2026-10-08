import { invoke } from "@tauri-apps/api/core";

export interface InvokeResult<T> {
  ok: boolean;
  data?: T;
  error?: string;
}

/**
 * Calls a Tauri command, returning { ok: false } instead of throwing when the
 * Rust backend is not wired yet. All stores use this so the UI renders with
 * placeholder/seed data before the backend agent implements the commands.
 */
export async function safeInvoke<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<InvokeResult<T>> {
  try {
    const data = await invoke<T>(command, args);
    return { ok: true, data };
  } catch (e) {
    console.warn(`[nexora] invoke "${command}" unavailable:`, e);
    return { ok: false, error: String(e) };
  }
}

/**
 * Typed wrappers around the real backend commands registered in
 * `src-tauri/src/main.rs` (one per `CAPABILITY_NOTES.md` entry).
 * Seed-data fallback is preserved: every wrapper returns
 * `{ ok: false }` (never throws) when the backend reports
 * `E_CORE_NOT_WIRED`, so stores/pages keep rendering placeholder data.
 */

// Hardware (docs/07)
export const hardwareDetect = <T>() => safeInvoke<T>("hardware_detect");

// Models + HF (docs/05, docs/04 §4.2)
export const analyzeModel = <T>(url: string) =>
  safeInvoke<T>("analyze_model", { url });
export const hfSearch = <T>(query: string) =>
  safeInvoke<T>("hf_search", { query });
export const modelFiles = <T>(model_id: string) =>
  safeInvoke<T>("model_files", { model_id });
export const modelLoad = (model_id: string) =>
  safeInvoke("model_load", { model_id });
export const modelUnload = (model_id: string) =>
  safeInvoke("model_unload", { model_id });

// Downloads (docs/08)
export const downloadStart = (model_id: string) =>
  safeInvoke("download_start", { model_id });
export const downloadPause = (model_id: string) =>
  safeInvoke("download_pause", { model_id });
export const downloadResume = (model_id: string) =>
  safeInvoke("download_resume", { model_id });
export const downloadCancel = (model_id: string) =>
  safeInvoke("download_cancel", { model_id });

// Runtimes (docs/02 §2.4, docs/04 §4.3)
export const runtimeInstall = (runtime_id: string) =>
  safeInvoke("runtime_install", { runtime_id });
export const runtimeStart = (runtime_id: string) =>
  safeInvoke("runtime_start", { runtime_id });
export const runtimeStop = (runtime_id: string) =>
  safeInvoke("runtime_stop", { runtime_id });
export const runtimeHealth = <T>(runtime_id: string) =>
  safeInvoke<T>("runtime_health", { runtime_id });

// Generation — UI calls prepare -> run ONLY (AGENTS.md rule 3).
export interface GenerationTrust {
  trust_level?: string;
  trust_remote_code?: boolean;
  has_custom_py?: boolean;
  /** "view-files" | "sandbox" | "cancel" — gate stays closed without "sandbox". */
  user_decision?: string;
}
export const generationPrepare = <T>(
  model_id: string,
  capability: string,
  params: Record<string, unknown>,
  trust?: GenerationTrust,
) => safeInvoke<T>("generation_prepare", { model_id, capability, params, ...trust });
export const generationRun = <T>(
  model_id: string,
  capability: string,
  params: Record<string, unknown>,
) => safeInvoke<T>("generation_run", { model_id, capability, params });

// Environments (docs/09)
export const envList = <T>() => safeInvoke<T>("env_list");
export const envReuseCheck = <T>() => safeInvoke<T>("env_reuse_check");
export const envRollback = <T>() => safeInvoke<T>("env_rollback");

// Workflows / jobs (docs/10)
export const workflowRun = (graph: unknown) =>
  safeInvoke("workflow_run", { graph: graph as Record<string, unknown> });

// Logs (docs/11 §11.6)
export const logsQuery = <T>(scope?: string, limit?: number) =>
  safeInvoke<T>("logs_query", { scope, limit });

// Settings (docs/04 §4.1)
export const settingsGetAll = <T>() => safeInvoke<T>("settings_get_all");
export const settingsSet = (key: string, value: unknown) =>
  safeInvoke("settings_set", { key, value });
