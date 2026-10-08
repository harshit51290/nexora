import { useState } from "react";
import { Link } from "react-router-dom";

/**
 * Maps machine-readable error codes (docs/04-DATA-MODEL.md §4.2) to human
 * cause + fix (docs/11-UI-UX.md §11.6). Beginners see this; raw logs hide
 * behind "View Technical Logs". No bare stack traces.
 */
const ERROR_MAP: Record<string, { cause: string; fix: string }> = {
  CUDA_OOM: {
    cause: "Not enough GPU memory for this model + settings.",
    fix: "Try CPU offload, lower precision (FP16 / Q4), smaller resolution, or a quantized model.",
  },
  VRAM_FIT: {
    cause: "This model does not fit your GPU (need more VRAM than you have).",
    fix: "Pick the recommended quantized variant, or run on CPU. Do not download the full-precision weights.",
  },
  CUDA_MISSING: {
    cause: "No CUDA-capable GPU / driver found.",
    fix: "Install the NVIDIA driver + CUDA runtime, or switch the runtime to CPU mode.",
  },
  DOWNLOAD_FAIL: {
    cause: "Download interrupted or the host refused the connection.",
    fix: "Resume the download. Check disk space and network, then retry.",
  },
  CHECKSUM_MISMATCH: {
    cause: "A downloaded file failed its checksum — it may be corrupt.",
    fix: "Re-download the affected files. If it repeats, the source revision changed.",
  },
  DISK_FULL: {
    cause: "Not enough free disk space.",
    fix: "Free space or move the data root to a larger drive in Settings.",
  },
  RUNTIME_CRASH: {
    cause: "The inference process exited unexpectedly (app is unaffected).",
    fix: "Check Runtime logs, restart the runtime, and retry with lower memory settings.",
  },
  MODEL_LOAD_FAIL: {
    cause: "The model failed to load into the runtime.",
    fix: "Verify the runtime matches the model type, then check file integrity (re-verify checksums).",
  },
  ENV_BUILD_FAIL: {
    cause: "The isolated Python environment failed to build.",
    fix: "Retry environment creation. Check the Environments tab for the failing package and pinned versions.",
  },
  UNSUPPORTED_MODEL: {
    cause: "The analyzer could not map this model to a supported runtime.",
    fix: "It may need an experimental backend or plugin. See the model Requirements tab.",
  },
  TRUST_BLOCK: {
    cause: "Blocked: this repo needs custom-code execution approval.",
    fix: "Review the code (View), run once in Sandbox, or Cancel. Never auto-trust remote code.",
  },
};

export function describeError(code: string): { cause: string; fix: string } {
  return (
    ERROR_MAP[code] ?? {
      cause: `Something went wrong (${code}).`,
      fix: "Check the technical logs below and retry. If it repeats, report the error code.",
    }
  );
}

export default function ErrorBanner({
  code,
  detail,
  onClose,
}: {
  code: string;
  detail?: string;
  onClose?: () => void;
}) {
  const [showTech, setShowTech] = useState(false);
  const { cause, fix } = describeError(code);
  return (
    <div className="rounded-xl border border-red-800 bg-red-950 p-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-red-200">Something needs attention</h3>
          <p className="mt-1 text-sm text-red-100">{cause}</p>
          <p className="mt-1 text-sm text-red-100">
            <span className="font-medium">Fix: </span>
            {fix}
          </p>
        </div>
        {onClose && (
          <button onClick={onClose} className="text-red-300 hover:text-white" aria-label="dismiss">
            ✕
          </button>
        )}
      </div>
      <div className="mt-2 flex gap-2 text-xs">
        <button
          onClick={() => setShowTech((v) => !v)}
          className="rounded bg-red-900 px-2 py-1 text-red-100 hover:bg-red-800"
        >
          {showTech ? "Hide Technical Logs" : "View Technical Logs"}
        </button>
        <Link
          to="/runtime"
          className="rounded bg-red-900 px-2 py-1 text-red-100 hover:bg-red-800"
        >
          Open Runtime Logs
        </Link>
      </div>
      {showTech && (
        <pre className="mt-2 max-h-40 overflow-auto rounded bg-black p-2 font-mono text-xs text-neutral-300">
          code: {code}
          {detail ? `\n${detail}` : "\n(no further detail captured)"}
        </pre>
      )}
    </div>
  );
}
