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
