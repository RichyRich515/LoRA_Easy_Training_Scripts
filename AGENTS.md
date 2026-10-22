# AGENTS.md

Notes for coding agents (and humans) maintaining this fork.

## Repository layout

Three nested git repos, each a fork of 67372a's repo; all fork work is on `moss-image` (the default branch of each fork).

| Path | Fork | Upstream (`origin`) | Upstream branch |
|---|---|---|---|
| `.` (PySide6 UI) | RichyRich515/LoRA_Easy_Training_Scripts | 67372a/LoRA_Easy_Training_Scripts | `refresh` |
| `backend/` (submodule, FastAPI server + installer) | RichyRich515/LoRA_Easy_Training_scripts_Backend | 67372a/LoRA_Easy_Training_scripts_Backend | `refresh` |
| `backend/sd_scripts/` (submodule, trainers) | RichyRich515/sd-scripts | 67372a/sd-scripts | `main` |

Remotes in each local repo: `origin` = 67372a (upstream), `fork` = RichyRich515 (push here). Both `.gitmodules`
files point at the forks with `branch = moss-image`.

## Workflow

- Commit inside-out: `backend/sd_scripts` first, then `backend` (commit the updated `sd_scripts` pointer), then the
  root repo (commit the updated `backend` pointer). Push each to `fork moss-image`.
- Never push to `origin` (67372a). Upstream contributions go through pull requests from the fork.
- Never commit venvs, training configs/TOMLs with local paths, or personal launch shortcuts.
- Commit messages end with a `Co-Authored-By:` line for the agent that wrote the change.

### Syncing upstream (67372a)

67372a is slow but active (commits every few weeks as of 2026-10). To pull their work, inside-out:

1. `backend/sd_scripts`: `git fetch origin && git merge origin/main`
2. `backend`: `git fetch origin && git merge origin/refresh`, then `git add sd_scripts && git commit`
3. root: `git fetch origin && git merge origin/refresh`, then `git add backend && git commit`
4. Push all three to `fork moss-image`. Keep the fork's `.gitmodules` URLs if upstream touches them.

MOSS work is mostly in new files, so conflicts should be rare. The shared files touched by the fork are listed below.

## Environment

- Backend venv: `backend/sd_scripts/venv` (Python 3.11 by default), built with uv by `backend/installer.py`:
  torch 2.14.1 + cu130 (cuDNN 9.24), torchvision 0.29.1, triton-windows 3.8, xformers 0.0.35, torchao 0.18. The
  flash-attn wheels were built for torch 2.7 and are disabled (`FLASH_ATTN_WHEELS = {}`).
- UI venv: `./venv`. The backend is started by the UI as `backend/sd_scripts/venv/Scripts/python.exe backend/main.py
  backend` and listens on port 8000.
- Run scripts with a venv's `python.exe` explicitly rather than a bare `python`.

## MOSS-Image training (sd-scripts)

Files: `moss_train_network.py`, `library/moss_image_models.py`, `moss_image_vae.py`, `moss_te_worker.py`,
`moss_te_client.py`, `moss_train_utils.py`, `strategy_moss.py`, `networks/lora_moss.py`, `docs/moss_train_network.md`.
Shared files touched: `networks/lora_anima.py` (network_class hook, behavior-neutral for Anima),
`library/train_util.py`, `library/sai_model_spec.py`; backend `main.py`, `utils/validation.py`, `installer.py`,
`custom_scheduler/.../low_bit_optim/*` (torchao 0.18 flags); UI `main_ui_files/MossUI.py`, `ui_files/MossUI.py`,
`ArgsListUI.py`, `MainUI.py`.

- The Qwen3.5 text encoder needs transformers 5 while the venv pins 4.x: it runs in a subprocess with an overlay in
  `backend/sd_scripts/moss_te_deps` (installed on first use, no torch inside, survives venv rebuilds).
- Upstream kohya sd-scripts has its own MOSS-Image trainer since 0.12.1 (2026-10-18, `moss_image_train_network.py`);
  useful as a reference, not merged here.

Verified findings (RTX 4090, Windows), keep in mind when changing things:
- cuDNN SDPA with cuDNN 9.7.1 (torch 2.7.1+cu128) returns NaN gradients for specific real inputs (deterministic,
  e.g. a 51x28 latent with a 128-token caption at sigma 0.97); cuDNN 9.24 is finite on the same inputs. Inductor
  also fails on cuDNN with torch 2.7.
- torch 2.7 ignores an `sdpa_kernel` context inside compiled code, so `set_attention_backend` also sets the
  process-wide SDPA priority (`torch._C._set_sdp_priority_order`).
- torch >= 2.8 inductor needs MSVC (`cl`) for dynamic shapes; `ensure_msvc_environment` loads vcvars64 via vswhere.
- INT8 (`--moss_int8`): ConvRot quantization bit-identical to upstream / ComfyUI `*_int8_convrot` files, matmul via
  cuBLAS `torch._int_mm` inside an autograd Function so inductor fuses (de)quantization. Upstream's Triton int8
  kernel is fast with Triton 3.3 but bf16-speed with Triton 3.8. Only faster with `--moss_compile blocks`.
- Speed at 1024² per image: torch 2.7.1 compiled 1.91 s; torch 2.14.1 + compile + cuDNN 1.42 s; + int8 1.19 s
  (7.5 GB peak). At 512² with int8, `moss_unchecked_blocks = 42` (no checkpointing) fits and is ~20% faster.
- int8 accuracy vs bf16: prediction error 2.9%, LoRA gradient cosine 0.72 (attention kernel swap alone: 0.66;
  fp8 weight-only: 0.59).

Testing: there is no MOSS test suite in the repo yet; scratch tests lived outside it (batched vs per-sample
forward, fused-file load, fused vs split LoRA, sampler, skip layers, fp32 modulation, int8 vs bf16, NaN repro).
A quick end-to-end check is a 20-step run with `--max_train_steps 20` on a cached dataset.
