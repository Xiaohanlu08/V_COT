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
- direct GitHub access is unavailable from the GPU server and source synchronization must continue through the Windows staging machine.
- `vcot` Conda environment was created successfully with Python 3.10.21.
- the server contains 10 x NVIDIA GeForce RTX 3090, each with 24 GiB VRAM.
- NVIDIA driver is 570.144 and `nvidia-smi` reports CUDA 12.8 capability.
- GPU 7 was occupied by another Python process using about 15.5 GiB during the probe; project development should avoid GPU 7 unless it becomes free.
- system `nvcc` is not currently available.
- PyTorch was intentionally not installed during the first compatibility probe.

The current driver is suitable for the pinned vLLM/PyTorch runtime path. The next controlled installation pins `torch==2.7.1`, `torchvision==0.22.1`, `vllm==0.10.0`, `transformers==4.54.0`, and `trl==0.15.2` before installing the remaining Monet requirements.

## Current Task
1. Synchronize the newly added `scripts/02_install_monet_runtime.sh` from GitHub to the Windows staging tree and upload it to the GPU server.
2. Run the script in the existing `vcot` environment using the TUNA PyPI mirror.
3. Verify package versions, `pip check`, CUDA availability, and a minimal CUDA tensor operation.

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
- [ ] Install and verify the pinned Monet runtime requirements.
- [ ] Download the official Monet-7B checkpoint through `hf-mirror.net`.
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
- Avoid GPU 7 while the currently observed external process remains active.
- Full-scale or RL/VLPO experiments may later use more 3090s or H200 when justified by memory/runtime.

This is a resource-allocation decision, not a change to the scientific goal.

## Active Method Version
None. V0 has not started.

## Current Experiment
Environment/runtime bootstrap only; not yet a scientific experiment.

## Known Issues
- Direct GitHub access from the target server is unavailable; GitHub must not be part of the server-side bootstrap path.
- Windows-to-Linux transfer previously converted shell scripts to CRLF; `.gitattributes` now enforces LF for shell scripts, and existing transferred scripts can be normalized with `sed -i 's/\r$//'` if needed.
- `nvcc` is not installed system-wide. This is not a blocker for the current prebuilt-wheel inference/runtime gate, but later DeepSpeed/custom CUDA extension compilation may require a CUDA toolkit.
- Official Monet SFT scripts are written for 8 GPUs with DeepSpeed ZeRO-2; they will not be treated as a drop-in 4 x RTX 3090 recipe.
- The project will not change torch/vLLM/Transformers versions casually after runtime verification.

## Next Action
Run `scripts/02_install_monet_runtime.sh`, return the complete terminal output, and do not download the Monet-7B checkpoint or run inference until the pinned runtime stack passes `pip check` and the CUDA tensor test.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
