# Current State

## Project Stage
Baseline reproduction / latent-mode verification.

## Current Objective
Establish a reproducible Monet baseline before implementing any new latent-supervision method.

## Current Branch
`main`

## Selected Upstream Baseline
- Monet repository: `https://github.com/NOVAglow646/Monet.git`
- Pinned commit: `08939998d3d643a73a316e349faa34f420429153`
- Primary checkpoint: `NOVAglow646/Monet-7B`
- Baseline specification: `BASELINE.md`

## Verified Infrastructure
Verified on 2026-09-12:
- Monet source is present at `~/work/V_COT/third_party/Monet` and HEAD is exactly `08939998d3d643a73a316e349faa34f420429153`.
- `vcot` Conda environment exists with Python 3.10.21.
- Server has 10 x RTX 3090, 24 GiB each; driver 570.144; `nvidia-smi` reports CUDA 12.8 capability.
- Verified runtime stack: `torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`, `vllm==0.10.0`, `transformers==4.54.0`, `trl==0.15.2`.
- CUDA is available and a CUDA tensor test passed.
- Monet-7B checkpoint is fully downloaded and structurally verified; four safetensors shards total 15.44 GiB, matching the checkpoint index.

## Verified Baseline Inference
The official Monet example runs successfully and returns the correct answer (`\\boxed{C}`). This marks official-example inference as reproduced.

## Latent-Mode Status
Natural latent activation is not yet verified:
1. the unmodified official example did not emit `<abs_vis_token>`;
2. a diagnostic prompt explicitly asking the model to begin with `<abs_vis_token>` also did not make the model emit it.

Tokenizer wiring is verified:
- `<abs_vis_token>` -> `151666`
- `</abs_vis_token>` -> `151667`
- `LATENT_SIZE=10`

A forced-token test successfully generated `[151666, 151666, 151666, 151666]`, but that run did **not** verify the Monet hidden-state path because the required spawn-safe `sitecustomize.py` was absent on the server. The expected `[VCOT_MONET_SITE]` / Monet-runner patch logs were also absent. Therefore that run must not be counted as a successful latent-runtime verification.

The server check confirmed:
- `~/work/V_COT/scripts/monet_site/sitecustomize.py` does not exist;
- `import sitecustomize` fails with `ModuleNotFoundError` under the intended `PYTHONPATH`.

This fully explains why Step 06 could force token 151666 yet still fail to prove that spawned vLLM workers were using Monet's custom `GPUModelRunner`.

## Current Task
Install the missing `scripts/monet_site/sitecustomize.py`, verify that Python resolves:
- `sitecustomize.__file__` to `scripts/monet_site/sitecustomize.py`;
- `vllm.v1.worker.gpu_model_runner.__file__` to `third_party/Monet/inference/vllm/monet_gpu_model_runner.py`.

Only after this lightweight preflight passes should Step 06 load Monet-7B again.

`scripts/06_force_latent_path.sh` has been hardened so it now:
- exits immediately if `sitecustomize.py` is missing;
- runs a preflight import check before model loading;
- asserts that the active vLLM runner comes from the Monet source tree.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [ ] Install and verify spawn-safe `sitecustomize.py` on the server.
- [ ] Verify the spawned worker is actually using the Monet custom runner.
- [ ] Exercise the latent hidden-state path with a deterministic forced-start diagnostic.
- [ ] Observe at least one natural latent-mode activation from the checkpoint, or document its trigger rate on an appropriate benchmark subset.
- [ ] Reproduce selected Monet benchmark baseline under documented settings.
- [ ] Freeze reproduced baseline with a Git tag.
- [ ] Locate and instrument the exact latent-state tensors needed for V0 experiments.
- [ ] Start V0 only after the above checks pass.

## Known Issues
- GitHub access from the GPU server is unavailable; source synchronization must use local staging or direct file handoff.
- Windows-to-Linux transfers may convert LF to CRLF; normalize transferred shell scripts before execution.
- `huggingface_hub` HEAD metadata calls are incompatible with the current HF mirror for this checkpoint; direct resumable GET is the verified workaround.
- `nvcc` is not installed system-wide. This is not a blocker for inference but may matter later for training extensions.
- vLLM shutdown may emit NCCL/resource-tracker warnings after successful inference; treat them as cleanup warnings unless they cause reproducible resource accumulation.
- Forced latent-token diagnostics are engineering tests only and must never be mixed with benchmark results.

## Next Action
Create `scripts/monet_site/sitecustomize.py` on the server, run the lightweight import preflight, and return its output. Do not reload Monet-7B until the preflight proves the Monet runner is active.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
