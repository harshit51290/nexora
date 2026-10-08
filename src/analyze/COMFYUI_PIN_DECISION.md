# ComfyUI Pin Decision (TODO-COMFYUI-URL resolution record)

Scope: `src/runtime/comfyui.rs` TODO-COMFYUI-URL items 1–3. This file is the
decision record only — `install()`/`prepare()` behavior is unchanged (still
`E-RUNTIME-NOT-INSTALLED` until the integrator confirms the pin with the
Windows smoke test below). Resolved 2026-10-08 against the live GitHub API
(no guessing; every URL below was returned by `api.github.com` that day).

## 1. Upstream repo

- Canonical source (matches `COMFYUI_REPO_URL` in `src/runtime/comfyui.rs`):
  `https://github.com/comfyanonymous/ComfyUI`
- Fetch method decision (TODO item 1): **`git clone` + `git checkout <tag>`**
  (pin-friendly, update-friendly — as the TODO already recommends). Portable
  `.7z` builds below are the documented fallback, not the primary path.

## 2. Tag to pin: `v0.39.0`

- Latest stable release as of 2026-10-08 (published 2026-10-05, not prerelease).
- Why this tag: newest stable — never float on `main` (ComfyUI breaks compat
  often); a dated, immutable tag keeps installs reproducible and rollback
  trivial (AGENTS.md rule 4).
- Status: **RECOMMENDED CANDIDATE — pending the Windows py3.11 smoke test**
  (TODO item 2). The integrator must run that test before `install()` fetches
  anything; on failure, walk back to the previous stable tag, not forward to
  `main`.

## 3. Portable fallback assets (exact URLs, `v0.39.0`)

Heads-up: release assets are hosted under `Comfy-Org/ComfyUI` (the API
redirects there from `comfyanonymous/ComfyUI`):

- NVIDIA (default CUDA):
  `https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_nvidia.7z`
- NVIDIA cu126:
  `https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_nvidia_cu126.7z`
- AMD:
  `https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_amd.7z`
- Intel:
  `https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_intel.7z`

Note: `v0.39.0` offers no cu121-labeled portable (only default + cu126). The
TODO's torch-cu121 reference predates this release — the smoke test decides
which portable (if any) matches our torch pin; do not assume cu126 works on
the 4GB Pascal reference target without running it.

## 4. Custom-node pack list (TODO item 3): UNDECIDED

No pack list is recorded here on purpose: the M13 video workflows that
determine the required nodes do not exist yet. Procedure: when M13 lands,
enumerate the node types used by generated workflows, pin each node repo to a
commit, and append the list to this file.

## 5. llama.cpp server builds (reference for the `llama_cpp` runtime)

llama.cpp ships rolling `bXXXX` tags (all marked prerelease) plus a `vX.Y`
stable marker that carries no server binaries — **pin an exact `bXXXX` tag,
never "latest"**. Latest `bXXXX` on 2026-10-08: **`b11510`**. Verified
Windows server assets (hosted under `ggml-org/llama.cpp`):

- CPU x64 (universal fallback):
  `https://github.com/ggml-org/llama.cpp/releases/download/b11510/llama-b11510-bin-win-cpu-x64.zip`
- CUDA 12.4 x64 (conservative pick for the 4GB Pascal reference target):
  `https://github.com/ggml-org/llama.cpp/releases/download/b11510/llama-b11510-bin-win-cuda-12.4-x64.zip`
- CUDA 13.4 x64 (newer cards only — verify before use):
  `https://github.com/ggml-org/llama.cpp/releases/download/b11510/llama-b11510-bin-win-cuda-13.4-x64.zip`

The archives contain `llama-server` (the binary the `llama_cpp` adapter
supervises). Re-resolve the newest `bXXXX` at integration time; b-tags move
daily.

## 6. Refresh / verification procedure

```powershell
# Newest stable ComfyUI tag:
(Invoke-WebRequest https://api.github.com/repos/comfyanonymous/ComfyUI/releases/latest -UseBasicParsing | ConvertFrom-Json).tag_name
# Newest llama.cpp build tag:
(Invoke-WebRequest https://api.github.com/repos/ggerganov/llama.cpp/tags?per_page=1 -UseBasicParsing | ConvertFrom-Json)[0].name
# Assets of a tag:
(Invoke-WebRequest https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/b11510 -UseBasicParsing | ConvertFrom-Json).assets |
  ForEach-Object { $_.browser_download_url }
```

1. Re-run the queries above. 2. Run the Windows py3.11 + torch smoke test
   against the candidate tag. 3. Only then update the pin + this file;
   `install()` keeps failing `E-RUNTIME-NOT-INSTALLED` until that happens.
