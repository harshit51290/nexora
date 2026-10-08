import type { CompatScore } from "../stores/models";

function Bar({ label, value }: { label: string; value: number }) {
  const color =
    value >= 75 ? "bg-emerald-500" : value >= 50 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 text-neutral-400">{label}</span>
      <div className="h-2 flex-1 rounded bg-neutral-800">
        <div className={`h-2 rounded ${color}`} style={{ width: `${value}%` }} />
      </div>
      <span className="w-10 text-right text-neutral-300">{value}%</span>
    </div>
  );
}

/**
 * Compat bars per docs/07-HARDWARE.md §7.4: overall / GPU / RAM / runtime %,
 * Good/Poor verdict + quantization recommendation. Shown BEFORE download/load
 * (AGENTS.md rule 6) — a Poor verdict blocks wasteful downloads.
 */
export default function CompatBars({
  compat,
  compact,
}: {
  compat: CompatScore;
  compact?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2 text-xs">
        <span className="font-medium">Compat {compat.overall}%</span>
        <span
          className={
            compat.verdict === "Good"
              ? "rounded bg-emerald-900 px-2 py-0.5 text-emerald-200"
              : "rounded bg-red-900 px-2 py-0.5 text-red-200"
          }
        >
          {compat.verdict}
        </span>
        {compat.quantRec && (
          <span className="text-neutral-400">rec: {compat.quantRec}</span>
        )}
      </div>
      {!compact && (
        <>
          <Bar label="GPU" value={compat.gpu} />
          <Bar label="RAM" value={compat.ram} />
          <Bar label="Runtime" value={compat.runtime} />
          {compat.requiredVramGb != null && (
            <p className="text-xs text-neutral-400">
              Needs ~{compat.requiredVramGb} GB VRAM.
            </p>
          )}
        </>
      )}
    </div>
  );
}
