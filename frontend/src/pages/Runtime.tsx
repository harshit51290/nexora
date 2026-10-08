import { useState } from "react";
import { useRuntimes } from "../stores/runtimes";
import { useSettings } from "../stores/settings";
import LogViewer from "../components/LogViewer";
import ErrorBanner from "../components/ErrorBanner";

/** Runtime supervisor UI: install / start / stop / health (docs/02 §2.4). */
export default function Runtime() {
  const { runtimes, install, start, stop, health } = useRuntimes();
  const { mode } = useSettings();
  const [healthMsg, setHealthMsg] = useState<string | null>(null);

  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <h2 className="text-xl font-semibold">Runtime</h2>
      {runtimes
        .filter((r) => r.status === "ERROR")
        .map((r) => (
          <ErrorBanner key={r.id} code={r.errorCode ?? "RUNTIME_CRASH"} />
        ))}
      {runtimes.map((r) => (
        <div key={r.id} className="card flex items-center justify-between gap-2">
          <div>
            <p className="font-medium">{r.kind}</p>
            <p className="text-xs text-neutral-400">
              {r.version ?? "unpinned"} · {r.status}
            </p>
          </div>
          <div className="flex gap-2 text-xs">
            {r.status === "NOT_INSTALLED" && (
              <button className="btn-primary" onClick={() => void install(r.id)}>
                Install
              </button>
            )}
            {(r.status === "READY" || r.status === "STOPPED") && (
              <button className="btn-primary" onClick={() => void start(r.id)}>
                Start
              </button>
            )}
            {r.status === "RUNNING" && (
              <button className="btn-secondary" onClick={() => void stop(r.id)}>
                Stop
              </button>
            )}
            <button
              className="btn-secondary"
              onClick={() =>
                void health(r.id).then((ok) =>
                  setHealthMsg(`${r.kind}: ${ok ? "healthy" : "no response (backend unwired?)"}`),
                )
              }
            >
              Health
            </button>
          </div>
        </div>
      ))}
      {healthMsg && <p className="text-sm text-neutral-300">{healthMsg}</p>}
      {mode === "developer" && (
        <div className="card text-sm text-neutral-400">
          Developer: runtime command lines, supervised child-process tree, and the Python console
          surface here (terminal passthrough arrives with the backend API/CLI milestone).
        </div>
      )}
      <LogViewer initialScope="Runtime" />
    </div>
  );
}
