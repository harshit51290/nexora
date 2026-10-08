"""llama.cpp shim — supervises the NATIVE `llama-server` binary.

NOTE (native binary first): inference runs in `runtimes/llama_cpp/bin/`
(`llama-server[.exe]`, per-OS release asset fetched by bootstrap.py). This
Python file is only the `--job`/HTTP adapter the Rust runner supervises
(`src/runtime/llama_cpp.rs` routes every request through here because the
binary speaks its own HTTP API, not our `--job` protocol).

  One-shot: serve.py --job '<json>'
    Spawns the binary on an ephemeral loopback port, waits for readiness,
    POSTs /completion, prints the generated text, terminates the child.
    Same exit codes as transformers/serve.py, plus: 3 = server binary
    missing (reinstall the runtime; see bootstrap.py SERVER_ASSETS TODO).

  Persistent server: serve.py serve [--host 127.0.0.1] [--port 8193]
    Keeps one binary child alive across /run calls (started lazily for the
    first model_dir seen); /health reflects child liveness; /shutdown (or
    SIGTERM/SIGINT) terminates the child gracefully.

Assumed `llama-server` surface (stable across recent releases; adjust here
if a vendored build differs): flags `--model <gguf> --host 127.0.0.1
--port <n>`, readiness = TCP accept on <n>, `POST /completion`
{"prompt","n_predict","temperature","seed"} -> {"content": ...} (tolerant
fallbacks for `choices[0].text` / `response` shapes included).

Job keys match Rust `JobPayload`. Supported params: max_tokens (default
256), temperature (default 0.7), n_gpu_layers (forwarded only when the
binary accepts `--n-gpu-layers`; default off = CPU, safest on 4GB).

Dependencies: stdlib only. No torch, no pip packages at all.
"""

import argparse
import json
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNTIME_ID = "llama_cpp"
DEFAULT_PORT = 8193

EXIT_OTHER = 1
EXIT_MODEL_MISSING = 2
EXIT_DEPS_MISSING = 3
EXIT_CUSTOM_CODE = 4
EXIT_CUDA_OOM = 5


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


def server_binary():
    here = Path(__file__).resolve().parent
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return here / "bin" / name


def find_gguf(model_dir):
    ggufs = sorted(model_dir.glob("*.gguf"))
    if not ggufs:
        fail_err(
            f"no .gguf weights in {model_dir}. Re-download the GGUF model.",
            EXIT_MODEL_MISSING,
        )
    return ggufs[0]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_port(port, timeout_s=120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def http_post(url, payload, timeout_s=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def completion_text(base_url, job):
    params = job.get("params") or {}
    body = {
        "prompt": job.get("prompt", ""),
        "n_predict": int(
            params.get("max_tokens", params.get("max_new_tokens", 256)) or 256
        ),
        "temperature": float(params.get("temperature", 0.7) or 0.7),
    }
    if job.get("seed") is not None:
        try:
            body["seed"] = int(job["seed"])
        except (TypeError, ValueError):
            pass
    data = http_post(base_url + "/completion", body)
    if isinstance(data.get("content"), str):
        return data["content"]
    choices = data.get("choices") or []
    if choices and isinstance(choices[0].get("text"), str):
        return choices[0]["text"]
    if isinstance(data.get("response"), str):  # older server shape
        return data["response"]
    raise ServeError(f"unexpected /completion response keys: {sorted(data)}")


class NativeServer:
    """Owns one `llama-server` child process (start / use / stop)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.proc = None
        self.base_url = None
        self.gguf = None

    def ensure(self, gguf, params):
        binary = server_binary()
        if not binary.is_file():
            fail_err(
                f"native server binary missing: {binary}. Reinstall the "
                "llama.cpp runtime from Environments.",
                EXIT_DEPS_MISSING,
            )
        with self._lock:
            if self.proc is not None and self.proc.poll() is None and self.gguf == gguf:
                return self.base_url
            self.stop_locked()
            port = free_port()
            cmd = [
                str(binary),
                "--model",
                str(gguf),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            try:
                n_layers = int((params or {}).get("n_gpu_layers", 0) or 0)
            except (TypeError, ValueError):
                n_layers = 0
            if n_layers > 0:
                cmd += ["--n-gpu-layers", str(n_layers)]
            eprint(f"spawning {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if not wait_for_port(port):
                tail = ""
                try:
                    _, tail = proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001 - best-effort diagnostics
                    proc.kill()
                fail_err(f"llama-server did not become ready on :{port}. {tail[-800:]}")
            self.proc = proc
            self.base_url = f"http://127.0.0.1:{port}"
            self.gguf = gguf
            threading.Thread(target=self._drain, args=(proc,), daemon=True).start()
            return self.base_url

    def _drain(self, proc):
        try:
            tail = proc.stderr.read() if proc.stderr else ""
            if tail:
                eprint(f"[llama-server] {tail[-2000:]}")
        except Exception:  # noqa: BLE001 - diagnostics only
            pass

    def stop_locked(self):
        proc, self.proc = self.proc, None
        self.base_url = None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def stop(self):
        with self._lock:
            self.stop_locked()


_SERVER = NativeServer()


def infer_text(job):
    raw = job.get("model_dir") or ""
    model_dir = Path(raw)
    if not raw or not model_dir.is_dir():
        fail_err(
            f"model directory not found: {raw!r}. Re-download the model.",
            EXIT_MODEL_MISSING,
        )
    gguf = find_gguf(model_dir)
    base_url = _SERVER.ensure(gguf, job.get("params") or {})
    try:
        return completion_text(base_url, job)
    except urllib.error.HTTPError as exc:
        fail_err(f"llama-server /completion failed: HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        fail_err(f"llama-server unreachable: {exc}")


def run_once(job):
    started = time.time()
    try:
        text = infer_text(job)
    except ServeError as exc:
        eprint(exc)
        _SERVER.stop()
        return exc.exit_code
    eprint(f"done in {int((time.time() - started) * 1000)}ms")
    print(text if text else "")
    _SERVER.stop()
    return 0


# --- persistent HTTP server -------------------------------------------------

_STATE = {"model": None, "lock": threading.Lock()}


class Handler(BaseHTTPRequestHandler):
    server_version = "NexoraServe/0.1"

    def log_message(self, *args):
        eprint(*args)

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
            alive = _SERVER.proc is not None and _SERVER.proc.poll() is None
            self._send(
                200,
                {
                    "status": "ok" if (model is None or alive) else "degraded",
                    "runtime": RUNTIME_ID,
                    "model": model,
                    "native_running": alive,
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
            started = time.time()
            try:
                text = infer_text(job)
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
                    "text": text,
                    "files": [],
                    "elapsed_ms": int((time.time() - started) * 1000),
                },
            )
        elif self.path == "/shutdown":
            self._send(200, {"status": "shutting down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send(404, {"error": "unknown path (try /run)"})


def cmd_serve(args):
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    def _stop(signum, _frame):
        eprint(f"signal {signum}: shutting down")
        _SERVER.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    eprint(
        f"serving on http://{args.host}:{server.server_port} (health + /run + /shutdown)"
    )
    try:
        server.serve_forever()
    finally:
        _SERVER.stop()
    eprint("stopped")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog=f"{RUNTIME_ID}/serve.py")
    ap.add_argument("--job", default=None, help="one-shot job JSON (Rust JobPayload)")
    sub = ap.add_subparsers(dest="command")
    srv = sub.add_parser("serve", help="persistent HTTP server")
    srv.add_argument("--host", default="127.0.0.1", help="bind host (never 0.0.0.0)")
    srv.add_argument("--port", type=int, default=DEFAULT_PORT)
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
    return run_once(payload)


if __name__ == "__main__":
    raise SystemExit(main())
