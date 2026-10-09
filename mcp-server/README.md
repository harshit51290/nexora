# Nexora MCP Server

Agent-facing control plane for Nexora (`mcp-server/`). Raw JSON-RPC 2.0 over
stdio — **no SDK dependency**, so it runs anywhere Python 3.10+ exists.

## Run it

```bash
python mcp-server/server.py            # stdio (attach from any MCP client)
python mcp-server/test_smoke.py        # 27 checks, no backend required
```

Claude Code / opencode client config (`mcp.json`):

```json
{
  "mcpServers": {
    "nexora": {
      "command": "python",
      "args": ["D:/code/universal ai runner/mcp-server/server.py"],
      "env": { "NEXORA_DATA_DIR": "D:/AI/Models" }
    }
  }
}
```

## Tools (9)

| Tool | Tonight? | How |
|---|---|---|
| `hardware_detect` | ✅ native | stdlib + `nvidia-smi` |
| `estimate_memory` | ✅ native | `hf_mem` engine, header-only Hub reads |
| `optimize_for_hardware` | ✅ native | detect + estimate + variant pick + verdict + flags + install cmd |
| `runtime_health` | ✅ native | probes shims :8191–8195 |
| `list_runtimes` | ✅ native | `runtimes/registry.json` from disk |
| `model_load` | ✅ router | dir check → runtime detect → ensure shim (spawns supervised, waits /health 30s) → record mapping |
| `model_unload` | ✅ router | drop mapping (shared shims stay up) |
| `models_loaded` | ✅ native | `state.json` mappings |
| `model_generate` | ✅ router | requires load; POSTs job to shim `/run`; surfaces coded shim errors |

Conventions: every failure is `{ok:false, code, error, fix}` — agents must
surface `fix` to users, never invent results. `optimize_for_hardware` fits =
weights + 25% headroom vs free VRAM (documented approximation; KV added
with `experimental`). State lives in `mcp-server/state.json` (gitignored
via `state.json` rule below — do NOT commit live mappings).

## When the Rust backend lands

`tools._api_post` is kept for future install/download-control tools that
will delegate to `http://127.0.0.1:8000` instead of inventing results.
Load/unload/generate stay on the router (it already works).
