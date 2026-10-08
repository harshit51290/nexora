import type { ModelEntry } from "../stores/models";
import { useModels } from "../stores/models";
import CompatBars from "./CompatBars";

function trustBadge(level: ModelEntry["trustLevel"]): string {
  switch (level) {
    case "verified":
      return "bg-emerald-900 text-emerald-200";
    case "community":
      return "bg-sky-900 text-sky-200";
    case "custom-code":
      return "bg-amber-900 text-amber-200";
    default:
      return "bg-neutral-800 text-neutral-300";
  }
}

/**
 * Model card per docs/11-UI-UX.md §11.3: name, task, params, VRAM, runtime,
 * [Run][Settings], compat %, license + trust badge.
 */
export default function ModelCard({ model }: { model: ModelEntry }) {
  const { select, toggleFavorite } = useModels();
  return (
    <div className="card flex flex-col gap-2">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold">{model.name}</h3>
          <p className="text-xs text-neutral-400">{model.repository}</p>
        </div>
        <button
          onClick={() => toggleFavorite(model.id)}
          aria-label="toggle favorite"
          className="text-lg text-neutral-500 hover:text-yellow-400"
        >
          {model.favorite ? "★" : "☆"}
        </button>
      </div>
      <div className="flex flex-wrap gap-1 text-xs">
        <span className="rounded bg-neutral-800 px-2 py-0.5">{model.task}</span>
        {model.params && (
          <span className="rounded bg-neutral-800 px-2 py-0.5">{model.params}</span>
        )}
        {model.vramGb != null && (
          <span className="rounded bg-neutral-800 px-2 py-0.5">
            {model.vramGb} GB VRAM
          </span>
        )}
        {model.runtime && (
          <span className="rounded bg-neutral-800 px-2 py-0.5">{model.runtime}</span>
        )}
        <span className="rounded bg-neutral-800 px-2 py-0.5">{model.status}</span>
      </div>
      {model.compat && <CompatBars compact compat={model.compat} />}
      <div className="flex items-center gap-2 text-xs">
        {model.license && (
          <span className="text-neutral-400">{model.license}</span>
        )}
        {model.trustLevel && (
          <span className={`rounded px-2 py-0.5 ${trustBadge(model.trustLevel)}`}>
            {model.trustLevel}
          </span>
        )}
      </div>
      <div className="mt-1 flex gap-2">
        <button
          className="btn-primary flex-1"
          onClick={() => select(model.id)}
        >
          Run
        </button>
        <button
          className="btn-secondary flex-1"
          onClick={() => select(model.id)}
        >
          Settings
        </button>
      </div>
    </div>
  );
}
