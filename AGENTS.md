# AGENTS.md — AI Builder Rules (read before coding)

1. Orchestration-only: never implement own NN/CUDA/diffusion/tensor engine; delegate to adapters.
2. Follow `docs/00-INDEX.md` build order; MVP = Transformers+Diffusers+llama.cpp, Text+Image, CUDA+CPU.
3. All runtimes implement `RuntimeAdapter` (`docs/06-RUNTIME-ADAPTERS.md`); UI calls `prepare->run` only.
4. Isolated envs per runtime; never global pip; pin Python/pkgs/CUDA/model rev; support rollback.
5. Security: `trust_remote_code`/custom `.py` always prompts View/Sandbox/Cancel; trust levels enforced; child-process isolation.
6. Check VRAM + compat estimate BEFORE download/load; 4GB profile is reference low-end.
7. Persist model/runtime state machines (`docs/04-DATA-MODEL.md`); every ERROR has code + human fix.
8. Keep docs↔code in sync; update registry + tasks when adding runtimes.
