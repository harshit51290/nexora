import { useState } from "react";
import { safeInvoke } from "../lib/tauri";

/** Job queue / batch + workflow templates (docs/10-API-CLI.md). */
export default function Workflows() {
  const [graph, setGraph] = useState('{\n  "steps": []\n}');
  const [queued, setQueued] = useState(false);

  async function enqueue() {
    let parsed: unknown = {};
    try {
      parsed = JSON.parse(graph);
    } catch {
      parsed = { _raw: graph };
    }
    await safeInvoke("workflow_run", { graph: parsed });
    setQueued(true);
  }

  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <h2 className="text-xl font-semibold">Workflows</h2>
      <p className="text-sm text-neutral-400">
        Chain models into batch jobs (e.g. LLM draft → TTS narration). Queued work runs through
        the scheduler with GPU-allocation checks.
      </p>
      <div className="card flex flex-col gap-3">
        <label className="block">
          <span className="label">Workflow graph (JSON)</span>
          <textarea
            className="input min-h-40 font-mono"
            value={graph}
            onChange={(e) => setGraph(e.target.value)}
          />
        </label>
        <button className="btn-primary self-start" onClick={() => void enqueue()}>
          Enqueue
        </button>
        {queued && (
          <p className="text-sm text-emerald-300">
            Queued (or recorded locally — job queue UI goes live once the backend wires
            workflow_run + job events).
          </p>
        )}
      </div>
    </div>
  );
}
