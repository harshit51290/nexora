import { useState } from "react";

export type LogScope = "Runtime" | "Download" | "Model" | "System";

export interface LogLine {
  ts: string;
  scope: LogScope;
  level: "INFO" | "WARN" | "ERROR";
  msg: string;
}

const PLACEHOLDER: LogLine[] = [
  {
    ts: new Date().toISOString(),
    scope: "System",
    level: "INFO",
    msg: "Log stream connects here once the backend wires logs_query / logs_subscribe.",
  },
];

/** Scoped log viewer per docs/11-UI-UX.md §11.6. */
export default function LogViewer({
  logs,
  initialScope,
}: {
  logs?: LogLine[];
  initialScope?: LogScope | "All";
}) {
  const [scope, setScope] = useState<LogScope | "All">(initialScope ?? "All");
  const lines = logs ?? PLACEHOLDER;
  const visible = scope === "All" ? lines : lines.filter((l) => l.scope === scope);
  return (
    <div className="card">
      <div className="mb-2 flex gap-1 text-xs">
        {(["All", "Runtime", "Download", "Model", "System"] as const).map((s) => (
          <button
            key={s}
            onClick={() => setScope(s)}
            className={
              scope === s
                ? "rounded bg-indigo-600 px-2 py-1 text-white"
                : "rounded bg-neutral-800 px-2 py-1 text-neutral-300 hover:bg-neutral-700"
            }
          >
            {s}
          </button>
        ))}
      </div>
      <pre className="max-h-64 overflow-auto rounded bg-black p-3 font-mono text-xs leading-relaxed">
        {visible.map((l, i) => (
          <div
            key={i}
            className={
              l.level === "ERROR"
                ? "text-red-400"
                : l.level === "WARN"
                  ? "text-amber-300"
                  : "text-neutral-300"
            }
          >
            [{l.ts}] [{l.scope}] {l.msg}
          </div>
        ))}
      </pre>
    </div>
  );
}
