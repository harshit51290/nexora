"""Diffusers inference server — SD / SDXL / Flux (+ image editing) pipelines.

Contract with the Rust child-process runner (`src/runtime/diffusers.rs`):

  One-shot (used by `prepare -> run` today):
    serve.py --job '<json>'
    stdout = log lines; the LAST line is the output image path (Rust parses
    the last line as the file).
    Same exit codes as transformers/serve.py (0 ok | 2 model | 3 deps |
    4 custom-code | 5 cuda-oom | 1 other).

  Persistent server: serve.py serve [--host 127.0.0.1] [--port 8192]
    GET  /health, POST /run (job json -> {"files":[...],"elapsed_ms":...}),
    POST /shutdown. Graceful on SIGTERM/SIGINT.

Job keys match Rust `JobPayload`. Supported params: steps (default 25),
guidance (default 7.0), height/width (default 512), seed, negative_prompt
(top-level or params), low_vram ("1" forces attention slicing + VAE tiling +
tiling-friendly fp16; auto-on when CUDA is present and params request it),
output_dir (default <cwd>/outputs/Images — Rust runs with cwd=data_dir).

Dependencies: stdlib only + diffusers/torch/Pillow (isolated env). Backend
imports are lazy so `--help` and `/health` work without torch installed.
"""

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNTIME_ID = "diffusers"
DEFAULT_PORT = 8192

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
    if not (path / "model_index.json").is_file():
        fail_err(
            f"not a diffusers layout (no model_index.json in {raw!r}). "
            "See the model page for the recommended runtime.",
            EXIT_MODEL_MISSING,
        )
    return path


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


def output_path(job, seed):
    params = job.get("params") or {}
    base = Path(params.get("output_dir") or (Path.cwd() / "outputs" / "Images"))
    base.mkdir(parents=True, exist_ok=True)
    slug = str(job.get("model_id") or "model").replace("/", "_")
    stamp = int(time.time())
    return base / f"{slug}-{seed}-{stamp}.png"


def infer_image(job):
    """Run one text-to-image job. Returns the saved image path as string."""
    model_dir = model_dir_of(job)
    params = job.get("params") or {}
    customs = sorted(p.name for p in model_dir.glob("*.py"))
    if customs and params.get("allow_custom_code") != "1":
        fail_err(
            "E-CUSTOM-CODE-BLOCKED: model ships custom code "
            f"({', '.join(customs)}); pass params.allow_custom_code=1 only "
            "after [View Files] + [Run in Sandbox] consent.",
            EXIT_CUSTOM_CODE,
        )
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        fail_err(
            f"diffusers/torch/Pillow not installed in this env: {exc}",
            EXIT_DEPS_MISSING,
        )
    use_cuda = torch.cuda.is_available()
    seed = job.get("seed")
    seed = coerce_int(seed, 0)
    steps = coerce_int(params.get("steps", 25), 25)
    guidance = coerce_float(params.get("guidance", 7.0), 7.0)
    height = coerce_int(params.get("height", 512), 512)
    width = coerce_int(params.get("width", 512), 512)
    negative = job.get("negative_prompt") or params.get("negative_prompt")
    low_vram = str(params.get("low_vram", "")).lower() in ("1", "true", "yes")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            trust_remote_code=False,
        )
        if use_cuda:
            pipe = pipe.to("cuda")
        if low_vram:
            pipe.enable_attention_slicing()
            try:
                pipe.enable_vae_tiling()
            except AttributeError:
                pass
        generator = torch.Generator(device="cuda" if use_cuda else "cpu").manual_seed(
            seed
        )
        image = pipe(
            job.get("prompt", ""),
            negative_prompt=negative,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        dest = output_path(job, seed)
        image.save(dest)
        return str(dest)
    except Exception as exc:  # noqa: BLE001 - mapped to coded exits below
        text = f"{type(exc).__name__}: {exc}"
        if classify_cuda_oom(text):
            fail_err(
                "CUDA out of memory. Enable low_vram=1, lower height/width/steps, "
                f"or use a smaller pipeline. ({text})",
                EXIT_CUDA_OOM,
            )
        fail_err(text)


def run_once(job):
    started = time.time()
    try:
        dest = infer_image(job)
    except ServeError as exc:
        eprint(exc)
        return exc.exit_code
    elapsed_ms = int((time.time() - started) * 1000)
    eprint(f"done in {elapsed_ms}ms -> {dest}")
    print(dest)  # last line = output path (Rust convention)
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
                dest = infer_image(job)
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
                    "files": [dest],
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
