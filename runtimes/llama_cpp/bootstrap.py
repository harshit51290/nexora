"""llama.cpp runtime bootstrap — fetches the native server, inits the shim env.

Usage (always launched by the Rust adapter as a child process):
    py -3 bootstrap.py --env-dir <environments/llama_cpp> install
    py -3 bootstrap.py --env-dir <environments/llama_cpp> check
    py -3 bootstrap.py --env-dir <environments/llama_cpp> pin --model-rev <rev>

Unlike the Python runtimes there is no torch stack here: inference runs in
the native `llama-server` binary (`runtimes/llama_cpp/bin/`, per-OS release
asset). The isolated venv exists only for the thin Python shim (`serve.py`,
quant recommendation helpers). NEVER global pip.
"""

import argparse
import json
import sys
import time
import urllib.request
import venv
from pathlib import Path

RUNTIME_ID = "llama_cpp"
PIN = {
    "python": "3.11",
    "packages": {},
    # Native binary tag, e.g. "b4926"; resolved at install time.
    "cuda": "cpu",
    "server_tag": None,
}

# TODO: fill per-OS release URLs when the Phase C3 (docs/12-BUILD-TASKS.md)
# binary-vendoring decision lands. Keep as data, not code branches elsewhere.
SERVER_ASSETS = {
    "win32": "llama-server.exe",
    "linux": "llama-server",
    "darwin": "llama-server",
}


def env_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "python" / "Scripts" / "python.exe"
    return env_dir / "python" / "bin" / "python"


def write_pin(env_dir: Path, model_rev: str | None) -> None:
    record = {
        "id": env_dir.name,
        "kind": RUNTIME_ID,
        "status": "ready",
        "created_at": int(time.time()),
        "pin": {**PIN, "model_rev": model_rev},
    }
    (env_dir / "environment.json").write_text(json.dumps(record, indent=2))


def cmd_install(env_dir: Path) -> int:
    target = env_dir / "python"
    if not env_python(env_dir).exists():
        print(f"[bootstrap:{RUNTIME_ID}] creating venv at {target}")
        venv.create(target, with_pip=True)
    # Binary fetch lands in runtimes/llama_cpp/bin/ (sibling of this file's
    # grandparent data root is unknown here; resolve relative to env dir).
    print(
        f"[bootstrap:{RUNTIME_ID}] NOTE: native server fetch not wired yet "
        f"(Phase C3). Venv shim ready; place {SERVER_ASSETS.get(sys.platform, 'llama-server')} "
        f"under runtimes/llama_cpp/bin/."
    )
    write_pin(env_dir, None)
    print(f"[bootstrap:{RUNTIME_ID}] ready (shim only)")
    return 0


def cmd_check(env_dir: Path) -> int:
    ok = env_python(env_dir).exists() and (env_dir / "environment.json").exists()
    print(("OK " if ok else "MISSING ") + str(env_dir))
    return 0 if ok else 1


def main() -> int:
    # urllib imported for the Phase C3 binary fetch; touch it so linters and
    # future wiring agree it is intentional.
    _ = urllib.request.urlopen
    ap = argparse.ArgumentParser(prog=f"{RUNTIME_ID}/bootstrap.py")
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--model-rev", default=None)
    ap.add_argument("command", choices=["install", "check", "pin"])
    args = ap.parse_args()
    env_dir = Path(args.env_dir)
    if args.command == "install":
        return cmd_install(env_dir)
    if args.command == "check":
        return cmd_check(env_dir)
    write_pin(env_dir, args.model_rev)  # pin
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
