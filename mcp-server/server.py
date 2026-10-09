"""Nexora MCP server — raw JSON-RPC 2.0 over stdio (no SDK dependency).

Speaks MCP: `initialize`, `tools/list`, `tools/call` (+ `ping`).
Each message is one JSON object per line. Run: `python mcp-server/server.py`.
"""

import json
import sys

import tools

TOOLS = [
    (
        "hardware_detect",
        "Detect CPU/RAM/GPU/VRAM/CUDA on this machine.",
        {"type": "object", "properties": {}},
    ),
    (
        "estimate_memory",
        "Pre-download weight/KV bytes for a Hub repo (hf-mem engine, no download).",
        {
            "type": "object",
            "required": ["model_id"],
            "properties": {
                "model_id": {"type": "string"},
                "revision": {"type": "string", "default": "main"},
                "experimental": {"type": "boolean", "default": False},
                "max_model_len": {"type": ["integer", "null"]},
                "batch_size": {"type": "integer", "default": 1},
                "kv_cache_dtype": {"type": "string", "default": "auto"},
                "gguf_file": {"type": ["string", "null"]},
            },
        },
    ),
    (
        "runtime_health",
        "Probe the five supervised runtime shims (:8191-8195).",
        {"type": "object", "properties": {}},
    ),
    (
        "list_runtimes",
        "Adapter ids + registry version from disk.",
        {"type": "object", "properties": {}},
    ),
    (
        "model_load",
        "Load a model: verify dir, detect runtime, ensure shim, record mapping.",
        {
            "type": "object",
            "required": ["model_id"],
            "properties": {
                "model_id": {"type": "string"},
                "runtime": {"type": ["string", "null"]},
                "cpu": {"type": "boolean", "default": False},
                "offload": {"type": "boolean", "default": False},
                "gpu_layers": {"type": ["integer", "null"]},
            },
        },
    ),
    (
        "model_unload",
        "Unload a model: drop the mapping (shared shims stay up).",
        {
            "type": "object",
            "required": ["model_id"],
            "properties": {"model_id": {"type": "string"}},
        },
    ),
    (
        "models_loaded",
        "List current model -> runtime mappings.",
        {"type": "object", "properties": {}},
    ),
    (
        "model_generate",
        "Generate through the model's live shim (needs model_load first).",
        {
            "type": "object",
            "required": ["model", "prompt"],
            "properties": {"model": {"type": "string"}, "prompt": {"type": "string"}},
        },
    ),
    (
        "optimize_for_hardware",
        "Full optimization: detect HW, estimate weights, pick fitting GGUF variant, verdict + flags + install command. Runnable tonight.",
        {
            "type": "object",
            "required": ["model_id"],
            "properties": {
                "model_id": {"type": "string"},
                "revision": {"type": "string", "default": "main"},
                "gguf_file": {"type": ["string", "null"]},
            },
        },
    ),
]

FUNCS = {name: getattr(tools, name) for name, _, _ in TOOLS}


def _text(payload):
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}


def handle(msg):
    """Pure dispatch (unit-testable without stdio). Returns response or None."""
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "nexora", "version": "0.1.0"},
            },
        }
    if method in ("notifications/initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "tools": [
                    {"name": n, "description": d, "inputSchema": s} for n, d, s in TOOLS
                ]
            },
        }
    if method == "tools/call":
        p = msg.get("params", {})
        name, args = p.get("name"), p.get("arguments", {}) or {}
        if name not in FUNCS:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32602, "message": f"unknown tool: {name}"},
            }
        try:
            return {"jsonrpc": "2.0", "id": mid, "result": _text(FUNCS[name](**args))}
        except TypeError as e:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32602, "message": f"bad arguments: {e}"},
            }
        except Exception as e:  # tools never raise by contract; belt + braces
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"},
            }
    return {
        "jsonrpc": "2.0",
        "id": mid,
        "error": {"code": -32601, "message": f"unknown method: {method}"},
    }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(json.loads(line))
        except json.JSONDecodeError as e:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"parse error: {e}"},
            }
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
