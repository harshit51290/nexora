import { create } from "zustand";
import { safeInvoke } from "../lib/tauri";

export type DownloadState =
  | "QUEUED"
  | "DOWNLOADING"
  | "PAUSED"
  | "VERIFYING"
  | "DONE"
  | "ERROR";

export interface DownloadEntry {
  modelId: string;
  bytesDone: number;
  bytesTotal: number;
  speedBps: number;
  etaSecs: number;
  state: DownloadState;
  errorCode?: string;
}

interface DownloadsState {
  downloads: Record<string, DownloadEntry>;
  start: (modelId: string) => Promise<void>;
  pause: (modelId: string) => Promise<void>;
  resume: (modelId: string) => Promise<void>;
  cancel: (modelId: string) => Promise<void>;
  applyProgress: (entry: DownloadEntry) => void;
}

export const useDownloads = create<DownloadsState>((set) => ({
  downloads: {},
  start: async (modelId) => {
    set((s) => ({
      downloads: {
        ...s.downloads,
        [modelId]: {
          modelId,
          bytesDone: s.downloads[modelId]?.bytesDone ?? 0,
          bytesTotal: s.downloads[modelId]?.bytesTotal ?? 0,
          speedBps: 0,
          etaSecs: 0,
          state: "QUEUED",
        },
      },
    }));
    await safeInvoke("download_start", { model_id: modelId });
  },
  pause: async (modelId) => {
    await safeInvoke("download_pause", { model_id: modelId });
    set((s) =>
      s.downloads[modelId]
        ? {
            downloads: {
              ...s.downloads,
              [modelId]: { ...s.downloads[modelId], state: "PAUSED" },
            },
          }
        : s,
    );
  },
  resume: async (modelId) => {
    await safeInvoke("download_resume", { model_id: modelId });
    set((s) =>
      s.downloads[modelId]
        ? {
            downloads: {
              ...s.downloads,
              [modelId]: { ...s.downloads[modelId], state: "DOWNLOADING" },
            },
          }
        : s,
    );
  },
  cancel: async (modelId) => {
    await safeInvoke("download_cancel", { model_id: modelId });
    set((s) => {
      const next = { ...s.downloads };
      delete next[modelId];
      return { downloads: next };
    });
  },
  // Called when the backend emits download-progress events (backend agent
  // wires the listener; see src-tauri/CAPABILITY_NOTES.md).
  applyProgress: (entry) =>
    set((s) => ({ downloads: { ...s.downloads, [entry.modelId]: entry } })),
}));

export function downloadPct(d: DownloadEntry): number {
  if (!d.bytesTotal) return 0;
  return Math.min(100, Math.round((d.bytesDone / d.bytesTotal) * 100));
}
