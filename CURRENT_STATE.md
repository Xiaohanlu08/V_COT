# Current State

## Project Stage
Baseline environment bootstrap / pre-inference.

## Current Objective
Establish a reproducible Monet baseline before implementing any new latent-supervision method.

## Current Branch
`main`

## Selected Upstream Baseline
- Monet repository: `https://github.com/NOVAglow646/Monet.git`
- Pinned commit: `08939998d3d643a73a316e349faa34f420429153`
- Primary checkpoint for first reproduction: `NOVAglow646/Monet-7B`
- Baseline specification: `BASELINE.md`

## Last Verified State
The V_COT project root and the pinned Monet source are present on the GPU server.

Verified on 2026-09-12:
- Monet source is at `~/work/V_COT/third_party/Monet`.
- server-side Monet HEAD is exactly `08939998d3d643a73a316e349faa34f420429153`.
- TUNA PyPI mirror is reachable (`HTTP/2 200`).
- `https://hf-mirror.net` is reachable (`HTTP/2 200`).
- direct GitHub access is unavailable from the GPU server and source synchronization must continue through the Windows staging machine or direct file handoff from ChatGPT.
- `vcot` Conda environment exists with Python 3.10.21.
- the physical server contains 10 x NVIDIA GeForce RTX 3090, each with 24 GiB VRAM.
- NVIDIA driver is 570.144; `nvidia-smi` reports CUDA 12.8 capability.
- system `nvcc` is not available, but it is not required for the current prebuilt-wheel inference gate.
- pinned Monet runtime installation completed successfully.
- verified runtime versions are:
  - `torch==2.7.1+cu126`
  - `torchvision==0.22.1+cu126`
  - `vllm==0.10.0`
  - `transformers==4.54.0`
  - `trl==0.15.2`
- PyTorch CUDA build is 12.6, CUDA is available, all 10 GPUs are visible, and a CUDA tensor operation passed on RTX 3090 (compute capability 8.6).

The runtime gate is therefore complete. The next gate is checkpoint download and official-example inference.

## Current Task
1. Run `scripts/03_download_monet7b.sh` to download `NOVAglow646/Monet-7B` through `hf-mirror.net` and verify the local checkpoint structure.
2. After successful download, run `scripts/04_monet_smoke_inference.sh` on a single RTX 3090 (physical GPU 0 by default).
3. Record the raw model output and whether `<abs_vis_token> ... </abs_vis_token>` appears, confirming latent-mode behavior.

## Next Milestones
- [x] Select the exact Monet implementation to use as the baseline.
- [x] Record the upstream repository URL and upstream commit SHA.
- [x] Characterize the server network as restricted-overseas rather than fully offline.
- [x] Verify TUNA PyPI availability from the server.
- [x] Verify HF mirror availability from the server.
- [x] Transfer the V_COT project-root files to the server.
- [x] Transfer pinned Monet source to `third_party/Monet` and verify its SHA on the server.
- [x] Create the `vcot` Python 3.10 environment through domestic mirrors.
- [x] Verify GPU driver/CUDA compatibility for the Monet/vLLM dependency set.
- [x] Install and verify the pinned Monet runtime requirements.
- [ ] Download and structurally verify the official Monet-7B checkpoint through `hf-mirror.net`.
- [ ] Reproduce official inference on at least one provided example.
- [ ] Observe/verify latent-mode generation behavior.
- [ ] Reproduce the selected Monet benchmark baseline under documented settings.
- [ ] Freeze the reproduced baseline with a Git tag.
- [ ] Locate the exact code path that creates/updates continuous latent visual states.
- [ ] Verify latent-state shape, positions, count, and generation behavior.
- [ ] Start V0 only after the above checks pass.

## Hardware Plan
- The physical server has 10 x RTX 3090.
- Development/debug/V0 pilot will initially use GPUs 0-3 as a 4 x RTX 3090 allocation.
- The first Monet smoke inference uses one physical RTX 3090, GPU 0 by default.
- Avoid GPU 7 while the previously observed external process remains active.
- Full-scale or RL/VLPO experiments may later use more 3090s or H200 when justified by memory/runtime.

This is a resource-allocation decision, not a change to the scientific goal.

## Active Method Version
None. V0 has not started.

## Current Experiment
Baseline checkpoint download and official-example smoke inference. This is still infrastructure/baseline reproduction, not a scientific experiment.

## Known Issues
- Direct GitHub access from the target server is unavailable; GitHub must not be part of the server-side bootstrap path.
- Windows GitHub access is also intermittent; when synchronization fails, current scripts can be handed off directly as files while GitHub remains the authoritative project record.
- Windows-to-Linux transfer previously converted shell scripts to CRLF; `.gitattributes` enforces LF for shell scripts, and transferred scripts should still be normalized with `sed -i 's/\r$//'` before execution when necessary.
- `nvcc` is not installed system-wide. This is not a blocker for inference, but later DeepSpeed/custom CUDA extension compilation may require a CUDA toolkit.
- Official Monet SFT scripts are written for 8 GPUs with DeepSpeed ZeRO-2; they will not be treated as a drop-in 4 x RTX 3090 recipe.
- The project will not change torch/vLLM/Transformers versions casually after runtime verification.

## Next Action
Download Monet-7B with `scripts/03_download_monet7b.sh`. Do not start benchmark evaluation yet. After structural verification succeeds, run `scripts/04_monet_smoke_inference.sh` and return the raw output plus latent-token check.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
