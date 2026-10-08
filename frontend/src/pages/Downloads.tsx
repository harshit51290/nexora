import { useDownloads, downloadPct } from "../stores/downloads";
import LogViewer from "../components/LogViewer";

/** Download manager UI (docs/08-DOWNLOAD-STORAGE.md). */
export default function Downloads() {
  const { downloads, pause, resume, cancel } = useDownloads();
  const list = Object.values(downloads);
  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <h2 className="text-xl font-semibold">Downloads</h2>
      {list.length === 0 ? (
        <p className="card text-sm text-neutral-400">
          Nothing downloading. Progress events stream here once the backend wires download_*
          commands.
        </p>
      ) : (
        list.map((d) => (
          <div key={d.modelId} className="card flex flex-col gap-2">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">{d.modelId}</span>
              <span className="text-neutral-400">
                {d.state} · {downloadPct(d)}%
              </span>
            </div>
            <div className="h-2 rounded bg-neutral-800">
              <div
                className="h-2 rounded bg-indigo-500"
                style={{ width: `${downloadPct(d)}%` }}
              />
            </div>
            <div className="flex gap-2 text-xs">
              {d.state === "DOWNLOADING" && (
                <button className="btn-secondary" onClick={() => void pause(d.modelId)}>
                  Pause
                </button>
              )}
              {d.state === "PAUSED" && (
                <button className="btn-secondary" onClick={() => void resume(d.modelId)}>
                  Resume
                </button>
              )}
              <button className="btn-secondary" onClick={() => void cancel(d.modelId)}>
                Cancel
              </button>
            </div>
          </div>
        ))
      )}
      <LogViewer initialScope="Download" />
    </div>
  );
}
