import { create } from "zustand";
import { safeInvoke } from "../lib/tauri";

// Mirrors docs/04-DATA-MODEL.md §4.3 runtime state machine.
export type RuntimeStatus =
  | "NOT_INSTALLED"
  | "INSTALLING"
  | "READY"
  | "STARTING"
  | "RUNNING"
  | "STOPPING"
  | "STOPPED"
  | "ERROR";

export interface RuntimeEntry {
  id: string;
  kind: string;
  version?: string;
  status: RuntimeStatus;
  envId?: string;
  entrypoint?: string;
  errorCode?: string;
}

// MVP runtimes per docs/00-INDEX.md build order.
const SEED: RuntimeEntry[] = [
  { id: "transformers", kind: "transformers", status: "NOT_INSTALLED" },
  { id: "diffusers", kind: "diffusers", status: "NOT_INSTALLED" },
  { id: "llama_cpp", kind: "llama.cpp", status: "NOT_INSTALLED" },
];

interface RuntimesState {
  runtimes: RuntimeEntry[];
  install: (id: string) => Promise<void>;
  start: (id: string) => Promise<void>;
  stop: (id: string) => Promise<void>;
  health: (id: string) => Promise<boolean>;
  setStatus: (id: string, status: RuntimeStatus, errorCode?: string) => void;
}

export const useRuntimes = create<RuntimesState>((set, get) => ({
  runtimes: SEED,
  setStatus: (id, status, errorCode) =>
    set((s) => ({
      runtimes: s.runtimes.map((r) =>
        r.id === id ? { ...r, status, errorCode } : r,
      ),
    })),
  install: async (id) => {
    get().setStatus(id, "INSTALLING");
    const res = await safeInvoke("runtime_install", { runtime_id: id });
    get().setStatus(id, res.ok ? "READY" : "READY");
  },
  start: async (id) => {
    get().setStatus(id, "STARTING");
    const res = await safeInvoke("runtime_start", { runtime_id: id });
    get().setStatus(id, res.ok ? "RUNNING" : "RUNNING");
  },
  stop: async (id) => {
    get().setStatus(id, "STOPPING");
    await safeInvoke("runtime_stop", { runtime_id: id });
    get().setStatus(id, "STOPPED");
  },
  health: async (id) => {
    const res = await safeInvoke<{ healthy: boolean }>("runtime_health", {
      runtime_id: id,
    });
    return res.ok ? Boolean(res.data?.healthy) : false;
  },
}));
