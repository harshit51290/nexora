#!/usr/bin/env python3
"""Example plugin entrypoint (docs/06 section 6.9).

Protocol: read one JSON payload from stdin, write one JSON result to stdout.
Real adapters (whisper/kokoro/ollama/custom) load their model here and run
inference in this child process — the host only supervises, never links in.
This stub echoes the contract so `spawn_plugin` has something to drive.
"""

import json
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 — protocol boundary, report as JSON
        json.dump(
            {"error": {"code": "E-PLUGIN-PROTOCOL", "message": str(exc)}}, sys.stdout
        )
        return 0
    json.dump(
        {
            "stub": True,
            "plugin": "example-tts",
            "received_keys": sorted(payload.keys()),
            "audio_path": None,
            "hint": "Replace this stub with the real adapter; keep the stdin/stdout JSON contract.",
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
