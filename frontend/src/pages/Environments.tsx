import { useState } from "react";
import { safeInvoke } from "../lib/tauri";

/** Isolated envs per runtime (docs/09-ENV-SECURITY.md). Never global pip. */
export default function Environments() {
  const [notice, setNotice] = useState<string | null>(null);

  async function action(cmd: string, msg: string) {
    await safeInvoke(cmd);
    setNotice(msg);
  }

  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <h2 className="text-xl font-semibold">Environments</h2>
      <p className="text-sm text-neutral-400">
        Each runtime gets an isolated environment with pinned Python / packages / CUDA / model
        revision. Envs are reused when pins match; rollback restores the previous pins.
      </p>
      <div className="card flex flex-wrap gap-2">
        <button className="btn-secondary" onClick={() => void action("env_list", "Env list requested — renders here once wired.")}>
          Refresh list
        </button>
        <button className="btn-secondary" onClick={() => void action("env_reuse_check", "Reuse check requested.")}>
          Check reuse
        </button>
        <button className="btn-secondary" onClick={() => void action("env_rollback", "Rollback requested.")}>
          Rollback
        </button>
      </div>
      {notice && <p className="text-sm text-emerald-300">{notice}</p>}
      <div className="card text-sm text-neutral-400">
        No environments yet. Installing a runtime creates its pinned env here.
      </div>
    </div>
  );
}
