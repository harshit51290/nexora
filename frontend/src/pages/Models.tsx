import { useState } from "react";
import { useModels } from "../stores/models";
import { useDownloads, downloadPct } from "../stores/downloads";
import ModelCard from "../components/ModelCard";
import CompatBars from "../components/CompatBars";
import LogViewer from "../components/LogViewer";
import ErrorBanner from "../components/ErrorBanner";

const FILTERS = ["All", "Text", "Image", "Audio", "Video", "Vision", "Installed", "Favorites"] as const;

const TABS = ["Overview", "Files", "Requirements", "Performance", "Versions", "Outputs", "Logs"] as const;

function matchFilter(task: string, caps: string[], filter: string): boolean {
  const t = `${task} ${caps.join(" ")}`.toLowerCase();
  switch (filter) {
    case "All":
      return true;
    case "Text":
      return t.includes("text") || t.includes("chat") || t.includes("llm");
    case "Image":
      return t.includes("image");
    case "Audio":
      return t.includes("audio") || t.includes("speech");
    case "Video":
      return t.includes("video");
    case "Vision":
      return t.includes("vision") || t.includes("ocr") || t.includes("detection");
    default:
      return true;
  }
}

/** Library + model detail tabs per docs/11-UI-UX.md §11.3. */
export default function Models() {
  const { models, selectedId, select, load, unload } = useModels();
  const { downloads, start } = useDownloads();
  const [filter, setFilter] = useState<string>("All");
  const [tab, setTab] = useState<string>("Overview");

  const visible = models.filter((m) => {
    if (filter === "Installed")
      return ["INSTALLED", "VALIDATING", "READY", "LOADED", "RUNNING"].includes(m.status);
    if (filter === "Favorites") return m.favorite;
    return matchFilter(m.task, m.capabilities, filter);
  });

  const selected = models.find((m) => m.id === selectedId) ?? null;
  const dl = selected ? downloads[selected.id] : undefined;

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-xl font-semibold">Models</h2>
      <div className="flex flex-wrap gap-1 text-xs">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={
              filter === f
                ? "rounded bg-indigo-600 px-2 py-1 text-white"
                : "rounded bg-neutral-800 px-2 py-1 text-neutral-300 hover:bg-neutral-700"
            }
          >
            {f}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          {visible.map((m) => (
            <ModelCard key={m.id} model={m} />
          ))}
          {visible.length === 0 && (
            <p className="text-sm text-neutral-400">No models match this filter.</p>
          )}
        </div>

        <div>
          {!selected ? (
            <p className="card text-sm text-neutral-400">
              Select a model (Run or Settings) to see details.
            </p>
          ) : (
            <div className="card flex flex-col gap-3">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold">{selected.name}</h3>
                  <p className="text-xs text-neutral-400">{selected.repository}</p>
                </div>
                <button onClick={() => select(null)} className="text-neutral-500 hover:text-white" aria-label="close details">
                  ✕
                </button>
              </div>

              {selected.status === "ERROR" && selected.errorCode && (
                <ErrorBanner code={selected.errorCode} />
              )}
              {selected.compat && <CompatBars compat={selected.compat} />}

              <div className="flex gap-2">
                {(selected.status === "READY" || selected.status === "INSTALLED") && (
                  <button className="btn-primary" onClick={() => void load(selected.id)}>
                    Load
                  </button>
                )}
                {selected.status === "LOADED" && (
                  <button className="btn-secondary" onClick={() => void unload(selected.id)}>
                    Unload
                  </button>
                )}
                {(selected.status === "SUPPORTED" || selected.status === "DISCOVERED") && (
                  <button className="btn-primary" onClick={() => void start(selected.id)}>
                    Download / Install
                  </button>
                )}
                {dl && (
                  <span className="self-center text-xs text-neutral-400">
                    {dl.state} {downloadPct(dl)}%
                  </span>
                )}
              </div>

              <div className="flex flex-wrap gap-1 border-t border-neutral-800 pt-2 text-xs">
                {TABS.map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={
                      tab === t
                        ? "rounded bg-indigo-600 px-2 py-1 text-white"
                        : "rounded bg-neutral-800 px-2 py-1 text-neutral-300 hover:bg-neutral-700"
                    }
                  >
                    {t}
                  </button>
                ))}
              </div>

              <div className="text-sm">
                {tab === "Overview" && (
                  <div className="flex flex-col gap-1 text-neutral-300">
                    <p>Task: {selected.task}</p>
                    <p>Runtime: {selected.runtime ?? "—"}</p>
                    <p>Capabilities: {selected.capabilities.join(", ") || "—"}</p>
                    <p>License: {selected.license ?? "—"}</p>
                    <p>Status: {selected.status}</p>
                  </div>
                )}
                {tab === "Files" && (
                  <p className="text-neutral-400">
                    File list loads via <code>model_files</code> once wired (sha256 + dedup hash per
                    docs/04 §4.1).
                  </p>
                )}
                {tab === "Requirements" && (
                  <p className="text-neutral-400">
                    VRAM, runtime, Python/CUDA pins and trust level (
                    {selected.trustLevel ?? "unknown"}) appear here. Custom code always prompts
                    View / Sandbox / Cancel.
                  </p>
                )}
                {tab === "Performance" && (
                  <p className="text-neutral-400">
                    Perf estimates + quant recommendation render here before download (docs/07
                    §7.4).
                  </p>
                )}
                {tab === "Versions" && (
                  <p className="text-neutral-400">
                    Revision pins + rollback targets list here (docs/09).
                  </p>
                )}
                {tab === "Outputs" && (
                  <p className="text-neutral-400">
                    Past generations with sidecar params (Reuse / Regenerate) list here.
                  </p>
                )}
                {tab === "Logs" && <LogViewer initialScope="Model" />}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
