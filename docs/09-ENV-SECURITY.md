# 09 — Environments & Security

## 9.1 Env manager (hardest component — isolate everything)
Different models need conflicting stacks (py3.10+torch2+cu12 vs py3.11+...). Never share one global env. Layout `environments/<id>/{python,packages,environment.json}`. Shared-runtime reuse: compatible models share one env (A+B+C -> Transformers env); incompatible -> separate envs via resolver. MVP may start 1-env-per-runtime-kind, then add resolver.

## 9.2 Reproducibility
Pin per env: Python/pkg/CUDA/driver/model revision (e.g. `Env#17: py3.11/torch2.x/transf5.x/cu12.x`). Rollback keeps prior env metadata; runtime update breaking model -> one-click rollback. Model updates optional, never silent. Periodic checks for model/runtime/plugin/app updates.

## 9.3 Containers
Docker/Podman = advanced opt-in later. Never mandatory on Windows. Native envs first.

## 9.4 Security (critical — HF can ship code)
- Never blind-exec repo code. `trust_remote_code=True` (or custom `.py`) triggers modal: `⚠ custom code — [View Files] [Run in Sandbox] [Cancel]`.
- Trust levels: Trusted (official/compatible) | Community | Unverified (odd code) | Blocked (dangerous/incompatible). Override allowed, never silent.
- Weight safety: downloaded `.bin/.pt/.pth/.ckpt/.pkl` get a picklescan-style opcode scan (`src/security/pickle.rs`); denylisted `GLOBAL` imports (process spawn, sockets, eval/exec) demote trust to Unverified. Safetensors is exempt by construction; `REDUCE`-alone never flags (torch uses it benignly).
- Isolation: every runtime = supervised child process (Rust/Python/ComfyUI); crash shows "Runtime crashed", app stays alive.
