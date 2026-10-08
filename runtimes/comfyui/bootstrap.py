"""ComfyUI runtime bootstrap — creates/refreshes the isolated env + server checkout.

Usage (always launched by the Rust adapter as a child process):
    py -3 bootstrap.py --env-dir <environments/comfyui> install
    py -3 bootstrap.py --env-dir <environments/comfyui> check
    py -3 bootstrap.py --env-dir <environments/comfyui> pin --model-rev <rev>

Rules (docs/09-ENV-SECURITY.md): stdlib only, NEVER global pip — every
`pip install` targets the env interpreter explicitly. Writes/refreshes
`environment.json` (pin record) next to the env.

TODO-COMFYUI-URL: pin the server checkout (COMFYUI_REPO_URL in
src/runtime/comfyui.rs) before M13; install clones pinned tag into
<env-dir>/server, then installs torch + requirements.
"""

import argparse
import json
import subprocess
import sys
import time
import venv
from pathlib import Path

RUNTIME_ID = "comfyui"
PIN = {
    "python": "3.11",
    "packages": {
        "torch": "2.3.1+cu121",
        "torchvision": "0.18.1+cu121",
        "torchaudio": "2.3.1+cu121",
    },
    "cuda": "cu121",
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
    py = str(env_python(env_dir))
    reqs = [f"{name}=={ver}" for name, ver in PIN["packages"].items()]
    print(f"[bootstrap:{RUNTIME_ID}] installing into isolated env (never global pip)")
    rc = subprocess.run(
        [py, "-m", "pip", "install", "--upgrade", "pip", *reqs],
        env={"__NEXORA_TORCH_INDEX__": "https://download.pytorch.org/whl/cu121"},
    ).returncode
    if rc != 0:
        print(f"[bootstrap:{RUNTIME_ID}] pip install failed (rc={rc})", file=sys.stderr)
        return rc
    # TODO-COMFYUI-URL: git clone --branch <pinned-tag> <COMFYUI_REPO_URL> <env-dir>/server
    write_pin(env_dir, None)
    print(f"[bootstrap:{RUNTIME_ID}] ready (server checkout pending TODO-COMFYUI-URL)")
    return 0


def cmd_check(env_dir: Path) -> int:
    ok = (
        env_python(env_dir).exists()
        and (env_dir / "environment.json").exists()
        and (env_dir / "server" / "main.py").exists()
    )
    print(("OK " if ok else "MISSING ") + str(env_dir))
    return 0 if ok else 1


def main() -> int:
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
