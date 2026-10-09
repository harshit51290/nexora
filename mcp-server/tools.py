"""Nexora MCP tool implementations — plain functions, no MCP imports.

Design rule (NIGHT-LOG.md D2): every tool is EITHER native-runnable tonight
(stdlib + `hf_mem`, both installed) OR a thin delegate to the Nexora API
(`NEXORA_API`, default http://127.0.0.1:8000) with a coded error when the
backend isn't up. No fakes: delegates never invent results.
"""

import json
import os
import platform
import shutil
import subprocess
import urllib.request

API_BASE = os.environ.get("NEXORA_API", "http://127.0.0.1:8000")
SHIM_PORTS = {
    "transformers": 8191,
    "diffusers": 8192,
    "llama_cpp": 8193,
    "audio": 8194,
    "comfyui": 8195,
}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_root():
    """Mirror `StorageLayout::default_root` (dirs::data_dir + /Nexora)."""
    override = os.environ.get("NEXORA_DATA_DIR")
    if override:
        return override
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share"
        )
    return os.path.join(base, "Nexora")


def _state_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


def _read_state():
    try:
        with open(_state_path(), encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(state):
    with open(_state_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _gb(b):
    return round(b / 1024 / 1024 / 1024, 2) if b is not None else None


# ---------------------------------------------------------------- native ---


def hardware_detect():
    """CPU/RAM/OS via stdlib + NVIDIA via nvidia-smi (None when absent)."""
    mem_total = mem_free = None
    try:  # Windows
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = (
                [("wLength", ctypes.c_ulong)]
                + [("x", ctypes.c_ulonglong) for _ in range(7)]
                + [("x2", ctypes.c_ulonglong)]
            )

        st = MS()
        st.wLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            mem_total = int(st.x) // (1024 * 1024)
    except Exception:
        pass
    if mem_total is None:
        try:  # POSIX fallback
            page = os.sysconf("SC_PAGE_SIZE") // 1024
            mem_total = os.sysconf("SC_PHYS_PAGES") * page // 1024
        except Exception:
            pass
    gpu, vram_total, vram_free, driver = None, None, None, None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if out.returncode == 0 and out.stdout.strip():
                name, total, free, drv = [
                    p.strip() for p in out.stdout.splitlines()[0].split(",")
                ]
                gpu, driver = name, drv
                vram_total = int(float(total))
                vram_free = int(float(free))
        except Exception:
            pass
    return {
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "ram_total_mb": mem_total,
        "gpu": gpu,
        "vram_total_mb": vram_total,
        "vram_free_mb": vram_free,
        "driver": driver,
        "cuda": gpu is not None,
        "os": platform.system(),
        "arch": platform.machine(),
    }


def estimate_memory(
    model_id,
    revision="main",
    experimental=False,
    max_model_len=None,
    batch_size=1,
    kv_cache_dtype="auto",
    gguf_file=None,
):
    """Weight/KV bytes WITHOUT downloading (hf_mem engine)."""
    from hf_mem import run as hf_run

    r = hf_run(
        model_id=model_id,
        revision=revision,
        experimental=experimental,
        max_model_len=max_model_len,
        batch_size=batch_size,
        kv_cache_dtype=kv_cache_dtype,
        gguf_file=gguf_file,
    )
    mem = r.memory if isinstance(r.memory, int) else None
    return {
        "model_id": r.model_id,
        "revision": r.revision,
        "filename": r.filename,
        "weights_bytes": mem,
        "weights_gb": _gb(mem),
        "per_file_gb": {k: _gb(v) for k, v in r.memory.items()}
        if isinstance(r.memory, dict)
        else None,
        "kv_bytes": r.kv_cache if isinstance(r.kv_cache, int) else None,
        "kv_gb": _gb(r.kv_cache) if isinstance(r.kv_cache, int) else None,
        "total_bytes": r.total_memory,
        "total_gb": _gb(r.total_memory),
        "experimental": experimental,
    }


def runtime_health():
    """Probe the five supervised shim servers (docs/06)."""
    out = {}
    for name, port in SHIM_PORTS.items():
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as res:
                out[name] = {
                    "up": True,
                    "status": res.status,
                    "body": json.loads(res.read().decode()),
                }
        except Exception as e:
            out[name] = {"up": False, "error": f"{type(e).__name__}: {e}"}
    return out


def list_runtimes():
    """Adapter registry + env pins, straight from disk."""
    with open(
        os.path.join(REPO_ROOT, "runtimes", "registry.json"), encoding="utf-8"
    ) as f:
        reg = json.load(f)
    ids = set()
    for m in reg.get("mappings", []):
        ids.update(m.get("runtimes", []))
    return {"adapters": sorted(ids), "registry_version": reg.get("version")}


def vram_tier(vram_total_mb):
    if vram_total_mb is None:
        return "cpu-only"
    if vram_total_mb < 4 * 1024:
        return "ultra-low"
    if vram_total_mb < 6 * 1024:
        return "low"
    if vram_total_mb < 12 * 1024:
        return "medium"
    if vram_total_mb < 24 * 1024:
        return "high"
    return "extreme"


def pick_variant(files_gb, free_vram_gb):
    """Largest variant with weights + 25% headroom fitting free VRAM.
    Mirror of `src/mem::pick_gguf_variant` (documented approximation)."""
    if free_vram_gb is None:
        return None
    for name in sorted(files_gb, key=files_gb.get, reverse=True):
        if files_gb[name] * 1.25 <= free_vram_gb:
            return name
    return None


def recommend_flags(runtime, tier):
    """Quant/dtype/offload family per docs/07 tier table."""
    rt = (runtime or "").lower().replace("-", "_")
    llama = "llama" in rt or "gguf" in rt
    diff = "diffus" in rt or "comfy" in rt
    if tier in ("cpu-only", "ultra-low", "low"):
        if llama:
            return {
                "dtype": "fp16",
                "quant": "Q4_K_M",
                "offload": True,
                "why": "4GB-class: Q4_K_M + partial GPU offload",
            }
        if diff:
            return {
                "dtype": "fp16",
                "quant": "none",
                "offload": True,
                "why": "SD-class on 4GB: fp16 + CPU offload + slicing/tiling",
            }
        return {
            "dtype": "fp16",
            "quant": "Q4_K_M",
            "offload": True,
            "why": "4GB-class: quantized weights + CPU offload",
        }
    quant = "Q4_K_M" if tier == "medium" else "Q8_0"
    return {
        "dtype": "fp16",
        "quant": quant if llama else ("none" if diff else "fp16"),
        "offload": False,
        "why": f"fits VRAM ({tier}); offload only if estimate exceeds free VRAM",
    }


def optimize_for_hardware(model_id, revision="main", gguf_file=None):
    """Full optimization bottom-up, runnable tonight: detect hardware,
    estimate weights, pick the fitting GGUF variant, verdict + flags +
    the exact install command. Mirrors docs/07 §7.4 + src/mem picker."""
    hw = hardware_detect()
    free_gb = (hw["vram_free_mb"] or hw["vram_total_mb"] or 0) / 1024
    tier = vram_tier(hw["vram_total_mb"])
    est = estimate_memory(model_id, revision=revision, gguf_file=gguf_file)
    files = est["per_file_gb"]
    runtimes = list_runtimes()["adapters"]
    if files:
        pick = pick_variant(files, free_gb) if free_gb else None
        smallest = min(files, key=files.get)
        runtime = "llama.cpp"
    else:
        pick, smallest, runtime = None, None, "diffusers/transformers"
    flags = recommend_flags(runtime, tier)
    need = (files[pick] if pick and files else (est["weights_gb"] or 0)) or 0
    fits = bool(free_gb and need * 1.25 <= free_gb)
    return {
        "hardware": {
            "gpu": hw["gpu"],
            "vram_gb": _gb((hw["vram_total_mb"] or 0) * 1024 * 1024),
            "tier": tier,
        },
        "estimate_gb": est["weights_gb"],
        "recommended_file": pick or smallest,
        "runtime": runtime,
        "verdict": "Good" if fits else ("Poor" if free_gb else "Unknown-hardware"),
        "flags": flags,
        "install": f"uar install {model_id}"
        + (f"  # then select file: {pick}" if pick else ""),
        "note": "Fits = weights + 25% headroom vs free VRAM (documented approximation; KV added when --experimental).",
    }


# ------------------------------------------------------------------ router ---
# The router handles the full model lifecycle for agents: resolve the model
# dir -> detect runtime -> ensure the shim is up (spawning it supervised if
# needed) -> record the mapping -> generate through shim /run.
# Shims are shared: unload drops the mapping but never kills a shim.


def _detect_runtime(model_dir):
    """File-marker detection (mirrors adapter detect() order)."""
    try:
        names = [p.name for p in __import__("pathlib").Path(model_dir).rglob("*")]
    except OSError:
        return "transformers"
    blob = "\n".join(names)
    if any(n.endswith(".gguf") for n in names):
        return "llama_cpp"
    if "model_index.json" in names:
        return "diffusers"
    if any(n.endswith(".onnx") for n in names):
        return "onnx"
    if "config_sentence_transformers.json" in names:
        return "audio"
    return "transformers"


def _shim_up(runtime):
    port = SHIM_PORTS.get(runtime)
    if port is None:
        return False
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=5
        ) as res:
            return res.status == 200
    except Exception:
        return False


def _spawn_shim(runtime):
    """Start a down shim as a supervised child; wait for /health (30s)."""
    port = SHIM_PORTS[runtime]
    serve = os.path.join(REPO_ROOT, "runtimes", runtime, "serve.py")
    if not os.path.exists(serve):
        return {
            "ok": False,
            "code": "E_RUNTIME_NOT_INSTALLED",
            "error": f"no serve.py for runtime '{runtime}'",
            "fix": "Install the runtime env first (`uar install` a model using it).",
        }
    import subprocess as _sp
    import time as _time

    try:
        proc = _sp.Popen(
            ["python", serve, "serve", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return {
            "ok": False,
            "code": "E_RUNTIME_CRASHED",
            "error": f"could not spawn shim: {e}",
            "fix": "Check python availability and the runtime env.",
        }
    deadline = __import__("time").time() + 30
    while __import__("time").time() < deadline:
        if _shim_up(runtime):
            state = _read_state()
            state.setdefault("_shims", {})[runtime] = proc.pid
            _write_state(state)
            return {"ok": True, "pid": proc.pid, "port": port}
        if proc.poll() is not None:
            return {
                "ok": False,
                "code": "E_RUNTIME_CRASHED",
                "error": f"shim for '{runtime}' exited during startup (code {proc.returncode})",
                "fix": "Check the runtime env (`bootstrap.py check`) and retry.",
            }
        _time.sleep(1)
    return {
        "ok": False,
        "code": "E_RUNTIME_NOT_READY",
        "error": f"shim for '{runtime}' did not answer /health in 30s",
        "fix": "Check port conflicts and retry.",
    }


def router_resolve(model_id, runtime=None):
    """Full path to a generatable model: dir check -> runtime -> live shim."""
    model_dir = os.path.join(_data_root(), "models", model_id)
    try:
        entries = os.listdir(model_dir)
    except OSError:
        entries = []
    if not entries:
        return {
            "ok": False,
            "code": "E_MODEL_NOT_FOUND",
            "error": f"no model files at {model_dir}",
            "fix": f"Install first: `uar install <owner/model>` naming it '{model_id}', or set NEXORA_DATA_DIR.",
        }
    rt = runtime or _detect_runtime(model_dir)
    if rt not in SHIM_PORTS:
        return {
            "ok": False,
            "code": "E_RUNTIME_NOT_FOUND",
            "error": f"runtime '{rt}' has no supervised shim",
            "fix": "Pick transformers/diffusers/llama_cpp/audio, or add a plugin.",
        }
    if not _shim_up(rt):
        started = _spawn_shim(rt)
        if not started.get("ok"):
            return started
    return {
        "ok": True,
        "model_id": model_id,
        "model_dir": model_dir,
        "runtime": rt,
        "port": SHIM_PORTS[rt],
    }


def model_load(model_id, runtime=None, cpu=False, offload=False, gpu_layers=None):
    """Load = verify dir + live shim + record mapping (state.json)."""
    resolved = router_resolve(model_id, runtime)
    if not resolved.get("ok"):
        return resolved
    state = _read_state()
    state[model_id] = {
        "runtime": resolved["runtime"],
        "model_dir": resolved["model_dir"],
        "prefs": {"cpu": cpu, "offload": offload, "gpu_layers": gpu_layers},
    }
    _write_state(state)
    return {
        "ok": True,
        "model_id": model_id,
        "status": "LOADED",
        "runtime": resolved["runtime"],
        "port": resolved["port"],
    }


def model_unload(model_id):
    """Unload = drop the mapping. Shared shims stay up for other models."""
    state = _read_state()
    if model_id not in state or model_id == "_shims":
        return {
            "ok": False,
            "code": "E_MODEL_NOT_LOADED",
            "error": f"'{model_id}' is not loaded",
            "fix": "Load it first with model_load.",
        }
    state.pop(model_id, None)
    _write_state(state)
    return {"ok": True, "model_id": model_id, "status": "READY"}


def models_loaded():
    """List current model -> runtime mappings."""
    return {k: v for k, v in _read_state().items() if k != "_shims"}


def model_generate(model, prompt, **params):
    """Generate through the model's live shim (requires model_load first)."""
    state = _read_state()
    entry = state.get(model)
    if not isinstance(entry, dict) or "runtime" not in entry:
        return {
            "ok": False,
            "code": "E_MODEL_NOT_LOADED",
            "error": f"'{model}' is not loaded",
            "fix": "Call model_load first, then generate.",
        }
    port = SHIM_PORTS[entry["runtime"]]
    job = {
        "model_dir": entry["model_dir"],
        "prompt": prompt,
        "negative_prompt": params.pop("negative_prompt", None),
        "seed": params.pop("seed", None),
        "params": params,
    }
    url = f"http://127.0.0.1:{port}/run"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(job).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as res:
            body = json.loads(res.read().decode())
            return {"ok": True, "model": model, "runtime": entry["runtime"], **body}
    except Exception as e:
        # Try to surface the shim's coded body if it sent one.
        detail = f"{type(e).__name__}: {e}"
        return {
            "ok": False,
            "code": "E_GENERATE_FAILED",
            "error": detail,
            "fix": "Check runtime_health; the model dir may lack weights or need custom-code consent.",
        }


# ------------------------------------------------- API delegate (future) -

# NOTE: load/unload/generate route through the shim router above (it works
# tonight). `_api_post` stays for the day the Rust API is up: future tools
# (e.g. install control, downloads) will delegate through it instead of
# inventing results (NIGHT-LOG D1).


def _api_post(path, payload):
    url = API_BASE + path
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as res:
            return {
                "ok": True,
                "status": res.status,
                "result": json.loads(res.read().decode()),
            }
    except Exception as e:
        return {
            "ok": False,
            "code": "E_BACKEND_DOWN",
            "error": f"{type(e).__name__}: {e}",
            "fix": f"Nexora API is not up. Start it (`cargo run --bin nexora` or `uar serve`), then retry {path}. Nothing was executed.",
        }
