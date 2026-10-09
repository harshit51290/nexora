"""Smoke tests for the Nexora MCP server. Run: python mcp-server/test_smoke.py.

Network tests (estimate) hit the Hub with header-only range reads (KBs).
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server
import tools

PASS = []


def check(name, cond, extra=""):
    assert cond, f"FAIL {name} {extra}"
    PASS.append(name)


# --- protocol framing (no I/O) ---
r = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
check("initialize", r["result"]["serverInfo"]["name"] == "nexora")
r = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
names = [t["name"] for t in r["result"]["tools"]]
for want in [
    "hardware_detect",
    "estimate_memory",
    "runtime_health",
    "list_runtimes",
    "model_load",
    "model_unload",
    "models_loaded",
    "model_generate",
    "optimize_for_hardware",
]:
    check(f"tool-{want}", want in names)
r = server.handle(
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "nope", "arguments": {}},
    }
)
check("unknown-tool", "error" in r)
r = server.handle(
    {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "hardware_detect", "arguments": {}},
    }
)
check("call-envelope", "result" in r and r["result"]["content"][0]["type"] == "text")

# --- native tools (no backend needed) ---
hw = tools.hardware_detect()
check("hw-shape", hw["cores"] and hw["os"] and "vram_total_mb" in hw, str(hw))
check(
    "tier-table",
    [tools.vram_tier(x) for x in [None, 1000, 4096, 8192, 16384, 32768]]
    == ["cpu-only", "ultra-low", "low", "medium", "high", "extreme"],
)
check(
    "picker",
    tools.pick_variant({"a": 8.0, "b": 3.0}, 6.0) == "b"
    and tools.pick_variant({"a": 8.0}, 16.0) == "a"
    and tools.pick_variant({"a": 8.0}, None) is None,
)
check(
    "flags",
    tools.recommend_flags("llama.cpp", "low")["quant"] == "Q4_K_M"
    and tools.recommend_flags("diffusers", "high")["offload"] is False,
)
rt = tools.list_runtimes()
check("registry", "llama_cpp" in rt["adapters"] and "diffusers" in rt["adapters"])
rh = tools.runtime_health()
check(
    "health-shape",
    set(rh) == {"transformers", "diffusers", "llama_cpp", "audio", "comfyui"},
)
down = tools.router_resolve("no-such-model-xyz")
check(
    "resolve-missing",
    down["ok"] is False
    and down["code"] == "E_MODEL_NOT_FOUND"
    and "uar install" in down["fix"],
)

# --- router round-trip on an isolated data root (fake weights dir) ---
import shutil
import tempfile

_fake_root = tempfile.mkdtemp(prefix="nexora-mcp-test-")
_fake_model = os.path.join(_fake_root, "models", "tiny-test")
os.makedirs(_fake_model)
with open(os.path.join(_fake_model, "model.gguf"), "wb") as f:
    f.write(b"GGUF-fake")
_state_backup = None
_state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
if os.path.exists(_state_file):
    with open(_state_file, encoding="utf-8") as f:
        _state_backup = f.read()
_old_env = os.environ.get("NEXORA_DATA_DIR")
os.environ["NEXORA_DATA_DIR"] = _fake_root
try:
    loaded = tools.model_load("tiny-test")
    check(
        "router-load",
        loaded["ok"] is True and loaded["runtime"] == "llama_cpp",
        str(loaded),
    )
    check("router-listed", "tiny-test" in tools.models_loaded())
    gen = tools.model_generate("tiny-test", "hello")
    check(
        "router-generate-surfaces-shim-error",
        gen["ok"] is False and gen["code"] == "E_GENERATE_FAILED",
        str(gen)[:200],
    )
    check("router-unload", tools.model_unload("tiny-test")["status"] == "READY")
    again = tools.model_unload("tiny-test")
    check(
        "router-unload-idempotent-guard",
        again["ok"] is False and again["code"] == "E_MODEL_NOT_LOADED",
    )
    gen2 = tools.model_generate("tiny-test", "hello")
    check(
        "router-generate-needs-load",
        gen2["ok"] is False and gen2["code"] == "E_MODEL_NOT_LOADED",
    )
finally:
    if _old_env is None:
        os.environ.pop("NEXORA_DATA_DIR", None)
    else:
        os.environ["NEXORA_DATA_DIR"] = _old_env
    if _state_backup is None:
        if os.path.exists(_state_file):
            os.remove(_state_file)
    else:
        with open(_state_file, "w", encoding="utf-8") as f:
            f.write(_state_backup)
    shutil.rmtree(_fake_root, ignore_errors=True)

# --- live estimate (header-only Hub reads) ---
est = tools.estimate_memory("gpt2")
check("est-gpt2", est["weights_bytes"] == 548090880, str(est["weights_bytes"]))
opt = tools.optimize_for_hardware("gpt2")
check(
    "opt-shape",
    opt["verdict"] in ("Good", "Poor", "Unknown-hardware")
    and opt["install"].startswith("uar install")
    and "flags" in opt,
)

print(f"SMOKE_OK {len(PASS)} checks")
