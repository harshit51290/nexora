import { useState } from "react";
import { safeInvoke } from "../lib/tauri";
import { useModels } from "../stores/models";
import ModelCard from "../components/ModelCard";

interface SearchHit {
  id: string;
  likes?: number;
}

/** HF search (docs/05-HF-INTEGRATION.md). Backend proxies the HF API. */
export default function Discover() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [searching, setSearching] = useState(false);
  const { analyze } = useModels();

  async function search() {
    setSearching(true);
    const res = await safeInvoke<SearchHit[]>("hf_search", { query });
    setHits(res.ok && res.data ? res.data : []);
    setSearching(false);
  }

  return (
    <div className="flex max-w-4xl flex-col gap-4">
      <h2 className="text-xl font-semibold">Discover</h2>
      <div className="flex gap-2">
        <input
          className="input"
          placeholder="Search HuggingFace models…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void search();
          }}
        />
        <button className="btn-primary shrink-0" disabled={searching} onClick={() => void search()}>
          {searching ? "Searching…" : "Search"}
        </button>
      </div>
      {hits.length === 0 ? (
        <p className="text-sm text-neutral-400">
          Results appear here once the backend wires <code>hf_search</code>. You can also paste any
          HF URL on Home.
        </p>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          {hits.map((h) => (
            <div key={h.id} className="card">
              <p className="font-medium">{h.id}</p>
              <button
                className="btn-secondary mt-2 text-xs"
                onClick={() => void analyze(`https://huggingface.co/${h.id}`)}
              >
                Analyze
              </button>
            </div>
          ))}
        </div>
      )}
      {/* Local mirror of analyzed models for convenience */}
      <ModelList />
    </div>
  );
}

import { useModels as useModelsList } from "../stores/models";
function ModelList() {
  const { models } = useModelsList();
  if (models.length === 0) return null;
  return (
    <div className="grid grid-cols-2 gap-3">
      {models.map((m) => (
        <ModelCard key={m.id} model={m} />
      ))}
    </div>
  );
}
