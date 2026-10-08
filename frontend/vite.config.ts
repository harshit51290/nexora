import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri dev server port must be fixed (see src-tauri/tauri.conf.json devUrl).
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    outDir: "dist",
    target: "es2021",
  },
});
