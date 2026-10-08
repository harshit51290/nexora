"""Transformers inference server — text generation (+classification/embedding passthrough).

Contract with the Rust child-process runner (`src/runtime/transformers.rs`):

  One-shot (used by `prepare -> run` today):
    serve.py --job '<json>'
    stdout = generated text (Rust takes stdout verbatim).
    stderr = diagnostics (Rust keeps the tail for E-RUNTIME-CRASHED mapping;
    a CUDA-OOM traceback keeps the "CUDA-OOM human hint" working end to end).
    exit 0 = ok | 2 = model missing | 3 = deps missing
         4 = custom-code blocked (E-CUSTOM-CODE-BLOCKED, needs trust consent)
         5 = cuda oom | 1 = other failure.

  Persistent server (future `server_binary` path; same shape as diffusers/audio):
    serve.py serve [--host 127.0.0.1] [--port 8191] [--model-dir D]
    GET  /health   -> {"status":"ok","runtime":"transformers","model":...}
    POST /run      <- job json -> {"text":...,"files":[],"elapsed_ms":...}
    POST /shutdown -> 200, then graceful stop (SIGTERM/SIGINT work too).

Job keys match Rust `JobPayload`: runtime, model_id, repository, revision,
model_dir, prompt, negative_prompt, seed, params. Supported params:
max_tokens (default 256), temperature (default 0.7), allow_custom_code ("1"
opts into trust_remote_code=True; default refuses custom repo code per
docs/09-ENV-SECURITY.md).

Dependencies: stdlib only + transformers/torch (provided by the isolated env;
never global pip). Backend imports are lazy so `--help` and `/health` work
without torch installed.
"""

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNTIME_ID = "transformers"
DEFAULT_PORT = 8191

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


def model_dir_of(job):
    raw = job.get("model_dir") or ""
    path = Path(raw)
    if not raw or not path.is_dir():
        fail_err(
            f"model directory not found: {raw!r}. Re-download the model.",
            EXIT_MODEL_MISSING,
        )
    return path


def check_custom_code(model_dir, params):
    """Refuse untrusted repo code unless explicitly consented (docs/09 §9.4)."""
    customs = sorted(p.name for p in model_dir.glob("*.py"))
    if customs and params.get("allow_custom_code") != "1":
        fail_err(
            "E-CUSTOM-CODE-BLOCKED: model ships custom code "
            f"({', '.join(customs)}); pass params.allow_custom_code=1 only "
            "after [View Files] + [Run in Sandbox] consent.",
            EXIT_CUSTOM_CODE,
        )


def coerce_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_cuda_oom(text):
    low = text.lower()
    return (
        ("out of memory" in low)
        or ("cuda error" in low)
        or ("cuda" in low and "alloc" in low)
    )


def infer_text(job):
    """Run one text-generation job. Returns the generated text."""
    model_dir = model_dir_of(job)
    params = job.get("params") or {}
    check_custom_code(model_dir, params)
    try:
        from transformers import pipeline, set_seed
    except ImportError as exc:
        fail_err(
            f"transformers/torch not installed in this env: {exc}", EXIT_DEPS_MISSING
        )
    seed = job.get("seed")
    if seed is not None:
        try:
            set_seed(int(seed))
        except (TypeError, ValueError):
            pass
    max_tokens = coerce_int(
        params.get("max_tokens", params.get("max_new_tokens", 256)), 256
    )
    temperature = coerce_float(params.get("temperature", 0.7), 0.7)
    try:
        gen = pipeline("text-generation", model=str(model_dir), trust_remote_code=False)
        out = gen(
            job.get("prompt", ""),
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
        )
        return out[0]["generated_text"]
    except Exception as exc:  # noqa: BLE001 - mapped to coded exits below
        text = f"{type(exc).__name__}: {exc}"
        if classify_cuda_oom(text):
            fail_err(
                "CUDA out of memory. Try CPU offload, FP16, a shorter "
                f"context, or a quantized variant. ({text})",
                EXIT_CUDA_OOM,
            )
        fail_err(text)


def run_once(job):
    started = time.time()
    try:
        text = infer_text(job)
    except ServeError as exc:
        eprint(exc)
        return exc.exit_code
    elapsed_ms = int((time.time() - started) * 1000)
    eprint(f"done in {elapsed_ms}ms")
    print(text if text else "")
    return 0


# --- persistent HTTP server -------------------------------------------------

_STATE = {"model": None, "lock": threading.Lock()}


class Handler(BaseHTTPRequestHandler):
    server_version = "NexoraServe/0.1"

    def log_message(self, *args):  # keep stdout clean; logs go to stderr
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
            self._send(200, {"status": "ok", "runtime": RUNTIME_ID, "model": model})
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
    if args.model_dir:
        with _STATE["lock"]:
            _STATE["model"] = args.model_dir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    def _stop(signum, _frame):
        eprint(f"signal {signum}: shutting down")
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
    # One-shot mode is a plain flag (current Rust argv: serve.py --job '<json>').
    ap.add_argument("--job", default=None, help="one-shot job JSON (Rust JobPayload)")
    sub = ap.add_subparsers(dest="command")
    srv = sub.add_parser("serve", help="persistent HTTP server")
    srv.add_argument("--host", default="127.0.0.1", help="bind host (never 0.0.0.0)")
    srv.add_argument("--port", type=int, default=DEFAULT_PORT)
    srv.add_argument("--model-dir", default=None)
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
