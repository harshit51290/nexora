import { useState } from "react";
import { useGeneration } from "../stores/generation";
import { useSettings } from "../stores/settings";

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      {children}
    </label>
  );
}

function NumberInput({
  value,
  onChange,
  step,
  min,
  max,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <input
      type="number"
      className="input"
      value={value}
      step={step}
      min={min}
      max={max}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

/**
 * Capability-driven generation form per docs/11-UI-UX.md §11.4 and
 * docs/04-DATA-MODEL.md §4.4. The schema derives from `capabilities`;
 * unknown capabilities fall back to a generic JSON form and never crash.
 */
export default function GenerateForm({
  modelId,
  capabilities,
}: {
  modelId: string;
  capabilities: string[];
}) {
  const { run, running } = useGeneration();
  const { mode } = useSettings();
  const [cap, setCap] = useState(capabilities[0] ?? "text-generation");

  // Text / chat
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2048);
  // Image
  const [negativePrompt, setNegativePrompt] = useState("");
  const [steps, setSteps] = useState(25);
  const [cfg, setCfg] = useState(7);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  // TTS
  const [voice, setVoice] = useState("af_heart");
  const [speed, setSpeed] = useState(1.0);
  // Generic fallback
  const [rawJson, setRawJson] = useState("{}");
  // Advanced (docs §11.5)
  const [dtype, setDtype] = useState("auto");
  const [quant, setQuant] = useState("auto");
  const [gpuLayers, setGpuLayers] = useState(-1);

  const known = [
    "text-generation",
    "chat",
    "text-to-image",
    "image-to-image",
    "text-to-speech",
  ];
  const isKnown = known.includes(cap);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    let params: Record<string, unknown> = {};
    if (cap === "text-generation" || cap === "chat") {
      params = { prompt, temperature, max_tokens: maxTokens };
    } else if (cap === "text-to-image" || cap === "image-to-image") {
      params = {
        prompt,
        negative_prompt: negativePrompt,
        steps,
        guidance_scale: cfg,
        width,
        height,
      };
    } else if (cap === "text-to-speech") {
      params = { text: prompt, voice, speed };
    } else {
      try {
        params = JSON.parse(rawJson || "{}");
      } catch {
        params = { _raw: rawJson };
      }
    }
    if (mode !== "beginner") {
      params = { ...params, dtype, quant, gpu_layers: gpuLayers };
    }
    await run(modelId, cap, params);
  }

  return (
    <form onSubmit={onSubmit} className="card flex flex-col gap-3">
      {capabilities.length > 1 && (
        <Field label="Capability">
          <select
            className="input"
            value={cap}
            onChange={(e) => setCap(e.target.value)}
          >
            {capabilities.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </Field>
      )}

      {(cap === "text-generation" || cap === "chat" || cap === "text-to-speech") && (
        <Field label={cap === "text-to-speech" ? "Text" : "Prompt"}>
          <textarea
            className="input min-h-24"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </Field>
      )}
      {(cap === "text-generation" || cap === "chat") && (
        <div className="grid grid-cols-2 gap-3">
          <Field label="Temperature">
            <NumberInput value={temperature} onChange={setTemperature} step={0.1} min={0} max={2} />
          </Field>
          <Field label="Max tokens">
            <NumberInput value={maxTokens} onChange={setMaxTokens} step={1} min={1} />
          </Field>
        </div>
      )}
      {(cap === "text-to-image" || cap === "image-to-image") && (
        <>
          <Field label="Prompt">
            <textarea
              className="input min-h-20"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </Field>
          <Field label="Negative prompt">
            <input
              className="input"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
            />
          </Field>
          <div className="grid grid-cols-4 gap-3">
            <Field label="Steps">
              <NumberInput value={steps} onChange={setSteps} min={1} max={150} />
            </Field>
            <Field label="CFG">
              <NumberInput value={cfg} onChange={setCfg} step={0.5} min={1} max={30} />
            </Field>
            <Field label="Width">
              <NumberInput value={width} onChange={setWidth} step={64} min={128} />
            </Field>
            <Field label="Height">
              <NumberInput value={height} onChange={setHeight} step={64} min={128} />
            </Field>
          </div>
        </>
      )}
      {cap === "text-to-speech" && (
        <div className="grid grid-cols-2 gap-3">
          <Field label="Voice">
            <input className="input" value={voice} onChange={(e) => setVoice(e.target.value)} />
          </Field>
          <Field label="Speed">
            <NumberInput value={speed} onChange={setSpeed} step={0.1} min={0.5} max={2} />
          </Field>
        </div>
      )}
      {!isKnown && (
        <Field label={`Generic params (unknown capability "${cap}")`}>
          <textarea
            className="input min-h-24 font-mono"
            value={rawJson}
            onChange={(e) => setRawJson(e.target.value)}
          />
        </Field>
      )}

      {mode !== "beginner" && (
        <details className="rounded-lg border border-neutral-700 p-3">
          <summary className="cursor-pointer text-sm font-medium">
            Advanced: runtime / precision / offload
          </summary>
          <div className="mt-3 grid grid-cols-3 gap-3">
            <Field label="dtype">
              <input className="input" value={dtype} onChange={(e) => setDtype(e.target.value)} />
            </Field>
            <Field label="quant">
              <input className="input" value={quant} onChange={(e) => setQuant(e.target.value)} />
            </Field>
            <Field label="gpu_layers (-1 = auto)">
              <NumberInput value={gpuLayers} onChange={setGpuLayers} min={-1} />
            </Field>
          </div>
        </details>
      )}

      <button type="submit" className="btn-primary" disabled={running || !modelId}>
        {running ? "Generating…" : cap === "text-to-speech" ? "Generate Audio" : "Generate"}
      </button>
    </form>
  );
}
