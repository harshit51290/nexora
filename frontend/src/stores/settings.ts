import { create } from "zustand";
import { safeInvoke } from "../lib/tauri";

export type UiMode = "beginner" | "advanced" | "developer";

interface SettingsState {
  /** Beginner is the default per docs/11-UI-UX.md §11.5. */
  mode: UiMode;
  dataRoot: string;
  autoUnload: boolean;
  setMode: (mode: UiMode) => void;
  setDataRoot: (path: string) => void;
  setAutoUnload: (v: boolean) => void;
  load: () => Promise<void>;
}

export const useSettings = create<SettingsState>((set, get) => ({
  mode: "beginner",
  dataRoot: "UniversalRunner",
  autoUnload: true,
  setMode: (mode) => {
    set({ mode });
    void safeInvoke("settings_set", { key: "ui.mode", value: mode });
  },
  setDataRoot: (path: string) => {
    set({ dataRoot: path });
    void safeInvoke("settings_set", { key: "storage.root", value: path });
  },
  setAutoUnload: (v: boolean) => {
    set({ autoUnload: v });
    void safeInvoke("settings_set", { key: "scheduler.auto_unload", value: v });
  },
  load: async () => {
    const res = await safeInvoke<Record<string, string>>("settings_get_all");
    if (res.ok && res.data) {
      const d = res.data;
      set({
        mode: (d["ui.mode"] as UiMode) ?? get().mode,
        dataRoot: d["storage.root"] ?? get().dataRoot,
      });
    }
  },
}));
