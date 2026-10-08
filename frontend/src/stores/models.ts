import { create } from "zustand";
import { safeInvoke } from "../lib/tauri";

// Mirrors docs/04-DATA-MODEL.md §4.2 model state machine.
export type ModelStatus =
  | "DISCOVERED"
  | "ANALYZING"
  | "SUPPORTED"
  | "DOWNLOADING"
  | "INSTALLED"
  | "VALIDATING"
  | "READY"
  | "LOADED"
  | "RUNNING"
  | "ERROR";

export type TrustLevel = "verified" | "community" | "untrusted" | "custom-code";

export interface CompatScore {
  overall: number;
  gpu: number;
  ram: number;
  runtime: number;
  verdict: "Good" | "Poor";
  quantRec?: string;
  requiredVramGb?: number;
}

export interface ModelEntry {
  id: string;
  name: string;
  repository: string;
  revision?: string;
  task: string;
  params?: string;
  vramGb?: number;
  runtime?: string;
  status: ModelStatus;
  capabilities: string[];
  license?: string;
  trustLevel?: TrustLevel;
  compat?: CompatScore;
  favorite?: boolean;
  lastUsed?: string;
  errorCode?: string;
}

// Seed data for Recently Used (docs/11-UI-UX.md §11.2) until backend wires up.
const SEED: ModelEntry[] = [
  {
    id: "qwen2.5-7b",
    name: "Qwen2.5 7B",
    repository: "Qwen/Qwen2.5-7B-Instruct",
    task: "text-generation",
    params: "7B",
    vramGb: 4.4,
    runtime: "llama.cpp",
    status: "READY",
    capabilities: ["text-generation", "chat"],
    license: "Apache-2.0",
    trustLevel: "verified",
    compat: { overall: 82, gpu: 75, ram: 90, runtime: 95, verdict: "Good", quantRec: "Q4_K_M", requiredVramGb: 4.4 },
    lastUsed: "2026-10-07",
  },
  {
    id: "sd15",
    name: "Stable Diffusion 1.5",
    repository: "runwayml/stable-diffusion-v1-5",
    task: "text-to-image",
    params: "0.9B",
    vramGb: 3.2,
    runtime: "diffusers",
    status: "READY",
    capabilities: ["text-to-image", "image-to-image"],
    license: "CreativeML Open RAIL-M",
    trustLevel: "verified",
    compat: { overall: 88, gpu: 85, ram: 92, runtime: 90, verdict: "Good", requiredVramGb: 3.2 },
    lastUsed: "2026-10-06",
  },
  {
    id: "whisper-small",
    name: "Whisper Small",
    repository: "openai/whisper-small",
    task: "automatic-speech-recognition",
    params: "0.2B",
    vramGb: 1.5,
    runtime: "transformers",
    status: "READY",
    capabilities: ["speech-to-text"],
    license: "MIT",
    trustLevel: "verified",
    compat: { overall: 95, gpu: 95, ram: 98, runtime: 95, verdict: "Good", requiredVramGb: 1.5 },
    lastUsed: "2026-10-05",
  },
  {
    id: "kokoro-82m",
    name: "Kokoro 82M",
    repository: "hexgrad/Kokoro-82M",
    task: "text-to-speech",
    params: "82M",
    vramGb: 0.8,
    runtime: "audio",
    status: "INSTALLED",
    capabilities: ["text-to-speech"],
    license: "Apache-2.0",
    trustLevel: "community",
    compat: { overall: 97, gpu: 98, ram: 99, runtime: 90, verdict: "Good", requiredVramGb: 0.8 },
    lastUsed: "2026-10-04",
  },
];

interface ModelsState {
  models: ModelEntry[];
  selectedId: string | null;
  analyzing: boolean;
  select: (id: string | null) => void;
  analyze: (url: string) => Promise<void>;
  setStatus: (id: string, status: ModelStatus, errorCode?: string) => void;
  toggleFavorite: (id: string) => void;
  markUsed: (id: string) => void;
  load: (id: string) => Promise<void>;
  unload: (id: string) => Promise<void>;
}

function repoFromUrl(url: string): string {
  const m = url.match(/huggingface\.co\/([^/\s]+\/[^/\s?#]+)/i);
  if (m) return m[1].replace(/\/$/, "");
  return url.trim();
}

export const useModels = create<ModelsState>((set, get) => ({
  models: SEED,
  selectedId: null,
  analyzing: false,
  select: (id) => set({ selectedId: id }),
  analyze: async (url) => {
    set({ analyzing: true });
    const repo = repoFromUrl(url);
    // Expected backend command (see src-tauri/CAPABILITY_NOTES.md).
    const res = await safeInvoke<ModelEntry>("analyze_model", { url });
    if (res.ok && res.data) {
      set((s) => ({ models: [res.data as ModelEntry, ...s.models], analyzing: false }));
      return;
    }
    // Fallback while backend is unwired: derive a DISCOVERED entry locally.
    const entry: ModelEntry = {
      id: repo.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
      name: repo.split("/")[1] ?? repo,
      repository: repo,
      task: "unknown",
      status: "ANALYZING",
      capabilities: [],
      trustLevel: "untrusted",
    };
    set((s) => ({ models: [entry, ...s.models], analyzing: false }));
  },
  setStatus: (id, status, errorCode) =>
    set((s) => ({
      models: s.models.map((m) =>
        m.id === id ? { ...m, status, errorCode } : m,
      ),
    })),
  toggleFavorite: (id) =>
    set((s) => ({
      models: s.models.map((m) =>
        m.id === id ? { ...m, favorite: !m.favorite } : m,
      ),
    })),
  markUsed: (id) =>
    set((s) => ({
      models: s.models.map((m) =>
        m.id === id
          ? { ...m, lastUsed: new Date().toISOString().slice(0, 10) }
          : m,
      ),
    })),
  load: async (id) => {
    const m = get().models.find((x) => x.id === id);
    if (!m) return;
    // AGENTS.md rule 6: compat/VRAM check happens BEFORE load — backend
    // enforces via vram manager; UI pre-gates on compat.verdict.
    if (m.compat && m.compat.verdict === "Poor") {
      get().setStatus(id, "ERROR", "VRAM_FIT");
      return;
    }
    await safeInvoke("model_load", { model_id: id });
    get().setStatus(id, "LOADED");
    get().markUsed(id);
  },
  unload: async (id) => {
    await safeInvoke("model_unload", { model_id: id });
    get().setStatus(id, "READY");
  },
}));
