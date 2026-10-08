import { useState } from "react";
import { Link } from "react-router-dom";
import { useModels } from "../stores/models";
import { useHardware, profileFor } from "../stores/hardware";

/** Extremely simple home per docs/11-UI-UX.md §11.2. */
export default function Home() {
  const { analyze, analyzing, models } = useModels();
  const { info, loading, refresh } = useHardware();
  const [url, setUrl] = useState("");
  const recent = [...models]
    .filter((m) => m.lastUsed)
    .sort((a, b) => (b.lastUsed ?? "").localeCompare(a.lastUsed ?? ""))
    .slice(0, 4);

  return (
    <div className="flex max-w-3xl flex-col gap-6">
      <section>
        <h2 className="text-xl font-semibold">Run AI models locally.</h2>
        <div className="mt-2 flex gap-2">
          <input
            className="input"
            placeholder="Paste HuggingFace URL, e.g. https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button
            className="btn-primary shrink-0"
            disabled={!url.trim() || analyzing}
            onClick={() => void analyze(url)}
          >
            {analyzing ? "Analyzing…" : "Analyze Model"}
          </button>
        </div>
      </section>

      <section className="card">
        <div className="flex items-center justify-between">
          <h3 className="font-medium">Hardware</h3>
          <button onClick={() => void refresh()} className="btn-secondary text-xs" disabled={loading}>
            {loading ? "Detecting…" : "Re-detect"}
          </button>
        </div>
        <p className="mt-2 text-sm">
          GPU {info.gpu} {info.vramGb}GB / RAM {info.ramGb}GB / CPU {info.cpu} /{" "}
          {info.cudaAvailable ? "CUDA available" : "CPU only"} / {info.storageFreeGb}GB free
        </p>
        <p className="mt-1 text-sm">
          Profile: {profileFor(info.vramGb)} · Status: {info.status}
        </p>
      </section>

      <section>
        <h3 className="mb-2 font-medium">Recently Used</h3>
        {recent.length === 0 ? (
          <p className="text-sm text-neutral-400">Nothing yet — analyze a model to begin.</p>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {recent.map((m) => (
              <Link key={m.id} to="/models" className="card hover:border-indigo-600">
                <p className="font-medium">{m.name}</p>
                <p className="text-xs text-neutral-400">
                  {m.task} · {m.status}
                </p>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
