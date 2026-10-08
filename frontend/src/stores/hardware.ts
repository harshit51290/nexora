import { create } from "zustand";
import { safeInvoke } from "../lib/tauri";

// Reference low-end target per docs/07-HARDWARE.md: GTX 1050 4GB / 16GB RAM.
export interface HardwareInfo {
  cpu: string;
  ramGb: number;
  gpu: string;
  vramGb: number;
  cudaAvailable: boolean;
  cudaVersion?: string;
  driverVersion?: string;
  storageFreeGb: number;
  os: string;
  arch: string;
  status: "Ready" | "Degraded" | "Unknown";
}

export type HardwareProfile =
  | "ultra-low"
  | "low"
  | "medium"
  | "high"
  | "extreme";

export function profileFor(vramGb: number): HardwareProfile {
  if (vramGb < 4) return "ultra-low";
  if (vramGb < 6) return "low";
  if (vramGb < 12) return "medium";
  if (vramGb < 24) return "high";
  return "extreme";
}

const PLACEHOLDER: HardwareInfo = {
  cpu: "Intel i7",
  ramGb: 16,
  gpu: "NVIDIA GTX 1050",
  vramGb: 4,
  cudaAvailable: true,
  storageFreeGb: 180,
  os: "Windows",
  arch: "x86_64",
  status: "Unknown",
};

interface HardwareState {
  info: HardwareInfo;
  loading: boolean;
  refresh: () => Promise<void>;
}

export const useHardware = create<HardwareState>((set) => ({
  info: PLACEHOLDER,
  loading: false,
  refresh: async () => {
    set({ loading: true });
    const res = await safeInvoke<HardwareInfo>("hardware_detect");
    if (res.ok && res.data) {
      set({ info: { ...res.data, status: "Ready" }, loading: false });
    } else {
      set({ loading: false });
    }
  },
}));
