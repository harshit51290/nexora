"""ComfyUI executor — queues API workflows on the ComfyUI server.

Fetch decision (scope item 6; mirrors `COMFYUI_REPO_URL` in
`src/runtime/comfyui.rs`):
  source = https://github.com/comfyanonymous/ComfyUI (`git clone`,
  recommended: pin + update friendly — NOT the portable zip).
  TODO-COMFYUI-URL: exact pin (release tag / commit) + custom-node pack list
  still undecided — the integrator must set the pin after the Windows
  py3.11 + torch-cu121 smoke test. Never float on `main`.

Contract with the Rust child-process runner:
  Today `src/runtime/comfyui.rs` launches `server/main.py` directly and this
  file is the adoption-ready executor (M13 rewires `run()` to it); its CLI
  already works standalone and for tests:

  One-shot: serve.py --job '<json>'
    params.workflow = staged workflow-JSON path (written by Rust
    `stage_workflow`), params.output_dir = where to collect outputs.
    stdout = one output file path per line. Same exit codes as
    transformers/serve.py (0 ok | 2 model/server missing | 3 deps | 1 other;
    ComfyUI is Apache-2.0 upstream code, no custom-code gate of ours here).

  Persistent server: serve.py serve [--host 127.0.0.1] [--port 8195]
    [--comfy-port 8188]
    GET /health (also reports whether the ComfyUI child is reachable),
    POST /run (job json -> {"files":[...],"elapsed_ms":...}),
    POST /shutdown. Graceful on SIGTERM/SIGINT; a ComfyUI child started by
    us is terminated, one we found already running is left alone.

ComfyUI HTTP surface used (stable prompt-queue API): POST /prompt
{"prompt": <workflow>, "client_id": ...} -> {"prompt_id": ...};
GET /history/<prompt_id> until done; GET /view?filename=..&subfolder=..&type=..
for each output. Server assumed at 127.0.0.1:<comfy-port>.

Dependencies: stdlib only.
"""

import argparse
import json
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNTIME_ID = "comfyui"
DEFAULT_PORT = 8195
DEFAULT_COMFY_PORT = 8188

# Scope item 6: decided source; pin decision still open (see header).
COMFYUI_REPO_URL = "https://github.com/comfyanonymous/ComfyUI"
# TODO-COMFYUI-URL: set the tested release tag/commit + node-pack list here
# and in bootstrap/install before fetching. Never float on `main`.
COMFYUI_PIN = None

EXIT_OTHER = 1
EXIT_MODEL_MISSING = 2
EXIT_DEPS_MISSING = 3


def eprint(*parts):
    print(f"[serve:{RUNTIME_ID}]", *parts, file=sys.stderr)


class ServeError(Exception):
    def __init__(self, message, exit_code=EXIT_OTHER):
        super().__init__(message)
        self.exit_code = exit_code


def fail(message, code=EXIT_OTHER):
    eprint(message)
    raise SystemExit(code)


def fail_err(message, code=EXIT_OTHER):
    raise ServeError(message, code)


def server_checkout():
    """runtimes/comfyui/server/main.py next to this file."""
    main_py = Path(__file__).resolve().parent / "server" / "main.py"
    return main_py if main_py.is_file() else None


def comfy_alive(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def http_json(method, url, payload=None, timeout_s=30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def http_bytes(url, timeout_s=600):
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read()


class ComfySupervisor:
    """Ensures a ComfyUI server is reachable (reuse or spawn)."""

    def __init__(self, port):
        self.port = port
        self._lock = threading.Lock()
        self.proc = None
        self.owned = False

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def ensure(self):
        with self._lock:
            if comfy_alive(self.port):
                return self.base_url
            main_py = server_checkout()
            if main_py is None:
                fail_err(
                    "ComfyUI server checkout missing "
                    f"(want {COMFYUI_REPO_URL} at runtimes/comfyui/server/; "
                    "TODO-COMFYUI-URL pin decision pending). Press Install on "
                    "the ComfyUI runtime card first.",
                    EXIT_MODEL_MISSING,
                )
            cmd = [
                sys.executable,
                str(main_py),
                "--listen",
                "127.0.0.1",
                "--port",
                str(self.port),
            ]
            eprint(f"spawning {' '.join(cmd)}")
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            self.owned = True
            deadline = time.time() + 300
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    tail = (
                        (self.proc.stderr.read() or "")[-2000:]
                        if self.proc.stderr
                        else ""
                    )
                    fail_err(f"ComfyUI server exited during startup. {tail}")
                if comfy_alive(self.port):
                    return self.base_url
                time.sleep(1.0)
            fail_err("ComfyUI server did not become ready in 300s.")

    def stop(self):
        with self._lock:
            proc, self.proc = self.proc, None
            self.owned = False
            if proc is None:
                return
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


def load_workflow(job):
    params = job.get("params") or {}
    raw = params.get("workflow") or ""
    path = Path(raw)
    if not raw or not path.is_file():
        fail_err(
            "params.workflow (staged workflow-JSON path) is missing. "
            "The scheduler must stage the workflow before queueing.",
            EXIT_MODEL_MISSING,
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail_err(f"cannot read workflow {path}: {exc}")
    # Accept both raw API-format graphs and {"prompt": {...}} wrappers.
    if isinstance(doc, dict) and "prompt" in doc and isinstance(doc["prompt"], dict):
        return doc["prompt"]
    if isinstance(doc, dict):
        return doc
    fail_err(f"workflow {path} is not a JSON object.")


def queue_and_collect(supervisor, job, timeout_s=1800):
    workflow = load_workflow(job)
    params = job.get("params") or {}
    client_id = f"nexora-{int(time.time() * 1000)}"
    try:
        accepted = http_json(
            "POST",
            supervisor.base_url + "/prompt",
            {"prompt": workflow, "client_id": client_id},
        )
        prompt_id = accepted["prompt_id"]
    except (urllib.error.URLError, OSError, KeyError) as exc:
        fail_err(f"cannot queue ComfyUI prompt: {exc}")
    deadline = time.time() + timeout_s
    outputs = []
    while time.time() < deadline:
        try:
            history = http_json("GET", f"{supervisor.base_url}/history/{prompt_id}")
        except (urllib.error.URLError, OSError) as exc:
            fail_err(f"ComfyUI history poll failed: {exc}")
        entry = history.get(prompt_id) or {}
        status = (
            (entry.get("status") or {}).get("status_str") == "success"
        ) or entry.get("outputs")
        if entry.get("outputs") or status:
            for _node, node_out in (entry.get("outputs") or {}).items():
                for item in node_out.get("images", []) + node_out.get("gifs", []):
                    outputs.append(item)
            if outputs or status:
                break
        time.sleep(2.0)
    if not outputs:
        fail_err(f"ComfyUI prompt {prompt_id} produced no outputs before timeout.")
    out_dir = Path(params.get("output_dir") or (Path.cwd() / "outputs" / "Images"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dests = []
    for item in outputs:
        query = urllib.parse.urlencode(
            {
                "filename": item.get("filename", ""),
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            }
        )
        blob = http_bytes(f"{supervisor.base_url}/view?{query}")
        dest = out_dir / Path(item.get("filename", f"{prompt_id}.png")).name
        dest.write_bytes(blob)
        dests.append(str(dest))
    return dests


_SUPERVISOR = None
_SUPERVISOR_LOCK = threading.Lock()


def supervisor_for(port):
    global _SUPERVISOR
    with _SUPERVISOR_LOCK:
        if _SUPERVISOR is None or _SUPERVISOR.port != port:
            if _SUPERVISOR is not None:
                _SUPERVISOR.stop()
            _SUPERVISOR = ComfySupervisor(port)
        return _SUPERVISOR


def run_once(job, comfy_port):
    started = time.time()
    supervisor = supervisor_for(comfy_port)
    try:
        supervisor.ensure()
        dests = queue_and_collect(supervisor, job)
    except ServeError as exc:
        eprint(exc)
        return exc.exit_code
    eprint(f"done in {int((time.time() - started) * 1000)}ms -> {len(dests)} file(s)")
    for dest in dests:
        print(dest)
    return 0


# --- persistent HTTP server -------------------------------------------------

_STATE = {"model": None, "lock": threading.Lock()}


class Handler(BaseHTTPRequestHandler):
    server_version = "NexoraServe/0.1"

    def log_message(self, *args):
        eprint(args[0] % args[1:])

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            with _STATE["lock"]:
                model = _STATE["model"]
            port = getattr(self.server, "comfy_port", DEFAULT_COMFY_PORT)
            self._send(
                200,
                {
                    "status": "ok",
                    "runtime": RUNTIME_ID,
                    "model": model,
                    "comfy_reachable": comfy_alive(port),
                },
            )
        else:
            self._send(404, {"error": "unknown path (try /health)"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", "replace") if length else "{}"
        if self.path == "/run":
            try:
                job = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._send(400, {"error": f"invalid job json: {exc}"})
                return
            port = getattr(self.server, "comfy_port", DEFAULT_COMFY_PORT)
            started = time.time()
            try:
                supervisor = supervisor_for(port)
                supervisor.ensure()
                dests = queue_and_collect(supervisor, job)
            except ServeError as exc:
                self._send(500, {"error": str(exc), "exit_code": exc.exit_code})
                return
            except Exception as exc:  # noqa: BLE001 - never crash the server
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            with _STATE["lock"]:
                _STATE["model"] = job.get("model_id")
            self._send(
                200,
                {
                    "text": None,
                    "files": dests,
                    "elapsed_ms": int((time.time() - started) * 1000),
                },
            )
        elif self.path == "/shutdown":
            self._send(200, {"status": "shutting down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send(404, {"error": "unknown path (try /run)"})


def cmd_serve(args):
    if args.host != "127.0.0.1" and args.host != "localhost":
        fail("refusing to bind non-loopback host (ComfyUI executes workflows).")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.comfy_port = args.comfy_port

    def _stop(signum, _frame):
        eprint(f"signal {signum}: shutting down")
        with _SUPERVISOR_LOCK:
            if _SUPERVISOR is not None:
                _SUPERVISOR.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    eprint(
        f"serving on http://{args.host}:{server.server_port} (health + /run + /shutdown)"
    )
    server.serve_forever()
    eprint("stopped")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog=f"{RUNTIME_ID}/serve.py")
    ap.add_argument("--job", default=None, help="one-shot job JSON (Rust JobPayload)")
    sub = ap.add_subparsers(dest="command")
    srv = sub.add_parser("serve", help="persistent HTTP server")
    srv.add_argument("--host", default="127.0.0.1", help="bind host (loopback only)")
    srv.add_argument("--port", type=int, default=DEFAULT_PORT)
    srv.add_argument("--comfy-port", type=int, default=DEFAULT_COMFY_PORT)
    args = ap.parse_args(argv)
    if args.command == "serve":
        return cmd_serve(args)
    if args.job is None:
        ap.print_help()
        return 2
    try:
        payload = json.loads(args.job)
    except json.JSONDecodeError as exc:
        fail(f"invalid job json: {exc}")
    return run_once(payload, args.comfy_port)


if __name__ == "__main__":
    raise SystemExit(main())
