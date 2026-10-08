import { useSettings } from "../stores/settings";

/** Settings: mode, relocatable data root, updates. */
export default function Settings() {
  const { mode, setMode, dataRoot, setDataRoot } = useSettings();
  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <h2 className="text-xl font-semibold">Settings</h2>
      <div className="card flex flex-col gap-3">
        <div>
          <span className="label">Experience mode (Beginner is default)</span>
          <div className="flex gap-2 text-sm">
            {(["beginner", "advanced", "developer"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={
                  mode === m
                    ? "rounded-lg bg-indigo-600 px-3 py-1.5 capitalize text-white"
                    : "rounded-lg bg-neutral-800 px-3 py-1.5 capitalize text-neutral-300 hover:bg-neutral-700"
                }
              >
                {m}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-neutral-400">
            Beginner: model → prompt → generate. Advanced: runtime / dtype / offload / quant / env.
            Developer: terminal, API, logs, env, model files.
          </p>
        </div>
        <label className="block">
          <span className="label">Data root (relocatable, e.g. D:\AI\Models)</span>
          <input
            className="input"
            value={dataRoot}
            onChange={(e) => setDataRoot(e.target.value)}
          />
        </label>
        <p className="text-xs text-neutral-500">
          Trust: custom code / trust_remote_code always prompts View / Sandbox / Cancel (AGENTS.md
          rule 5). No setting disables that prompt.
        </p>
      </div>
    </div>
  );
}
