import { Link } from "react-router-dom";
import { useModels } from "../stores/models";
import { useGeneration } from "../stores/generation";
import GenerateForm from "../components/GenerateForm";
import ErrorBanner from "../components/ErrorBanner";

/** Unified capability-driven generation (docs/11-UI-UX.md §11.4). */
export default function Generate() {
  const { models } = useModels();
  const { history, lastErrorCode, clearError } = useGeneration();
  const ready = models.filter((m) =>
    ["READY", "LOADED", "RUNNING"].includes(m.status),
  );
  const first = ready[0] ?? null;

  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <h2 className="text-xl font-semibold">Generate</h2>
      {lastErrorCode && <ErrorBanner code={lastErrorCode} onClose={clearError} />}
      {ready.length === 0 ? (
        <p className="card text-sm text-neutral-400">
          No ready model. <Link to="/models" className="text-indigo-400 underline">Install one</Link>{" "}
          first — compat is checked before download.
        </p>
      ) : (
        <>
          <p className="text-sm text-neutral-400">
            Model: <span className="font-medium text-white">{first?.name}</span> (
            {first?.capabilities.join(", ")})
          </p>
          {first && (
            <GenerateForm
              key={first.id}
              modelId={first.id}
              capabilities={first.capabilities}
            />
          )}
        </>
      )}

      <h3 className="mt-2 font-medium">History</h3>
      {history.length === 0 ? (
        <p className="text-sm text-neutral-400">No generations yet.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {history.map((h) => (
            <div key={h.id} className="card text-sm">
              <p className="font-medium">
                {h.modelId} · {h.capability}
              </p>
              <p className="text-xs text-neutral-400">{h.createdAt}</p>
              {h.outputText && <p className="mt-1">{h.outputText}</p>}
              {h.outputPath && (
                <p className="mt-1 text-neutral-300">Output: {h.outputPath}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
