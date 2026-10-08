import { useHardware, profileFor } from "../stores/hardware";
import { useSettings } from "../stores/settings";

/** Hardware dashboard (docs/07-HARDWARE.md). No NVIDIA hardcode — CUDA is one backend. */
export default function Hardware() {
  const { info, loading, refresh } = useHardware();
  const { autoUnload, setAutoUnload } = useSettings();
  const profile = profileFor(info.vramGb);

  const rows: [string, string][] = [
    ["CPU", info.cpu],
    ["System RAM", `${info.ramGb} GB`],
    ["GPU", `${info.gpu} (${info.vramGb} GB VRAM)`],
    ["CUDA", info.cudaAvailable ? `available${info.cudaVersion ? ` (${info.cudaVersion})` : ""}` : "not detected — CPU mode"],
    ["Driver", info.driverVersion ?? "—"],
    ["Storage free", `${info.storageFreeGb} GB`],
    ["OS / arch", `${info.os} / ${info.arch}`],
    ["Profile", `${profile} (4GB is the reference low-end target)`],
    ["Status", info.status],
  ];

  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Hardware</h2>
        <button onClick={() => void refresh()} className="btn-secondary text-xs" disabled={loading}>
          {loading ? "Detecting…" : "Re-detect"}
        </button>
      </div>
      <div className="card">
        <table className="w-full text-sm">
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k} className="border-b border-neutral-800 last:border-0">
                <td className="py-1.5 pr-4 text-neutral-400">{k}</td>
                <td className="py-1.5">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <label className="card flex cursor-pointer items-center justify-between text-sm">
        <span>
          Auto-unload other models when VRAM is short
          <span className="block text-xs text-neutral-400">
            Model A loaded + B requested + VRAM short → unload A → load B (docs/07 §7.5).
          </span>
        </span>
        <input
          type="checkbox"
          checked={autoUnload}
          onChange={(e) => setAutoUnload(e.target.checked)}
          className="h-4 w-4"
        />
      </label>
    </div>
  );
}
