"""Audio inference server — Whisper STT + TTS/playback routing (docs/06 §6.6).

Contract with the Rust child-process runner (`src/runtime/audio.rs`):

  One-shot: serve.py --job '<json>'
    stdout = transcript text (STT), or the output audio path on the LAST line
    (TTS — Rust treats a last line ending in .wav/.mp3/.ogg/.flac as a file).
    Same exit codes as transformers/serve.py.

  Persistent server: serve.py serve [--host 127.0.0.1] [--port 8194]
    GET /health, POST /run (job json -> {"text":...} or {"files":[...]}),
    POST /shutdown. Graceful on SIGTERM/SIGINT.

The Rust adapter injects params.audio_task ("SpeechToText" | "TextToSpeech" |
...); an explicit params.audio_path selects the STT input file, otherwise the
prompt is treated as the audio path. TTS writes 16-bit PCM WAV via stdlib
`wave` (no soundfile dependency); playback stays a UI concern.

Dependencies: stdlib only + transformers/torch (+ numpy for TTS output).
Backend imports are lazy so `--help` and `/health` work without torch.
"""

import argparse
import json
import signal
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNTIME_ID = "audio"
DEFAULT_PORT = 8194
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac")

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


def classify_cuda_oom(text):
    low = text.lower()
    return (
        ("out of memory" in low)
        or ("cuda error" in low)
        or ("cuda" in low and "alloc" in low)
    )


def coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_stt(job, model_dir, params):
    try:
        from transformers import pipeline
    except ImportError as exc:
        fail_err(
            f"transformers/torch not installed in this env: {exc}", EXIT_DEPS_MISSING
        )
    audio_in = params.get("audio_path") or job.get("prompt", "")
    if not audio_in or not Path(audio_in).is_file():
        fail_err(
            f"STT needs an audio file: params.audio_path is missing ({audio_in!r}).",
            EXIT_MODEL_MISSING,
        )
    asr = pipeline(
        "automatic-speech-recognition", model=str(model_dir), trust_remote_code=False
    )
    out = asr(audio_in)
    return out.get("text", "") if isinstance(out, dict) else str(out)


def infer_tts(job, model_dir, params):
    try:
        from transformers import pipeline
    except ImportError as exc:
        fail_err(
            f"transformers/torch not installed in this env: {exc}", EXIT_DEPS_MISSING
        )
    try:
        import numpy as np
    except ImportError:
        fail_err("TTS output needs numpy in the isolated env.", EXIT_DEPS_MISSING)
    text = job.get("prompt", "")
    if not text:
        fail_err("TTS needs prompt text to speak.", EXIT_OTHER)
    speed = coerce_float(params.get("speed", 1.0), 1.0)
    tts = pipeline("text-to-speech", model=str(model_dir), trust_remote_code=False)
    out = tts(text)
    audio = np.asarray(out["audio"], dtype=np.float32).ravel()
    rate = int(out.get("sampling_rate", 22050))
    if speed != 1.0:  # naive resample for speed; quality tradeoff documented
        idx = (np.arange(int(len(audio) / speed)) * speed).astype(int)
        idx = idx[idx < len(audio)]
        audio = audio[idx]
    pcm = (max(-1.0, min(1.0, float(x))) for x in audio)
    frames = b"".join(int(s * 32767).to_bytes(2, "little", signed=True) for s in pcm)
    base = Path(params.get("output_dir") or (Path.cwd() / "outputs" / "Audio"))
    base.mkdir(parents=True, exist_ok=True)
    slug = str(job.get("model_id") or "model").replace("/", "_")
    dest = base / f"{slug}-{int(time.time())}.wav"
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return str(dest)


def infer_audio(job):
    """Returns (text_or_None, files). STT -> transcript; TTS -> [wav path]."""
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
    task = str(params.get("audio_task", "SpeechToText"))
    try:
        if task.lower().startswith("texttospeech"):
            return None, [infer_tts(job, model_dir, params)]
        return infer_stt(job, model_dir, params), []
    except ServeError:
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to coded exits below
        text = f"{type(exc).__name__}: {exc}"
        if classify_cuda_oom(text):
            fail_err(f"CUDA out of memory. ({text})", EXIT_CUDA_OOM)
        fail_err(text)


def run_once(job):
    started = time.time()
    try:
        text, files = infer_audio(job)
    except ServeError as exc:
        eprint(exc)
        return exc.exit_code
    eprint(f"done in {int((time.time() - started) * 1000)}ms")
    if files:
        print(files[-1])  # last line = audio path (Rust convention)
    else:
        print(text or "")
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
                text, files = infer_audio(job)
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
                    "files": files,
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
