import { create } from "zustand";
import { safeInvoke } from "../lib/tauri";

export interface GenerationRecord {
  id: string;
  modelId: string;
  capability: string;
  params: Record<string, unknown>;
  outputPath?: string;
  outputText?: string;
  createdAt: string;
  runtime?: string;
  seed?: number;
}

interface GenerationState {
  history: GenerationRecord[];
  running: boolean;
  lastErrorCode: string | null;
  /** AGENTS.md rule 3: UI only ever calls prepare -> run, never the runtime directly. */
  run: (
    modelId: string,
    capability: string,
    params: Record<string, unknown>,
  ) => Promise<GenerationRecord | null>;
  clearError: () => void;
}

export const useGeneration = create<GenerationState>((set) => ({
  history: [],
  running: false,
  lastErrorCode: null,
  run: async (modelId, capability, params) => {
    set({ running: true, lastErrorCode: null });
    const prep = await safeInvoke<{ ok: boolean; error_code?: string }>(
      "generation_prepare",
      { model_id: modelId, capability, params },
    );
    if (prep.ok && prep.data && prep.data.ok === false) {
      set({ running: false, lastErrorCode: prep.data.error_code ?? "UNKNOWN" });
      return null;
    }
    const res = await safeInvoke<GenerationRecord>("generation_run", {
      model_id: modelId,
      capability,
      params,
    });
    if (!res.ok || !res.data) {
      // Backend not wired yet — record a placeholder so history UI works.
      const record: GenerationRecord = {
        id: `gen-${Date.now()}`,
        modelId,
        capability,
        params,
        createdAt: new Date().toISOString(),
        outputText: "[backend not wired — no output yet]",
      };
      set((s) => ({ history: [record, ...s.history], running: false }));
      return record;
    }
    set((s) => ({ history: [res.data as GenerationRecord, ...s.history], running: false }));
    return res.data as GenerationRecord;
  },
  clearError: () => set({ lastErrorCode: null }),
}));
