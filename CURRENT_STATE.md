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
- Server has 10 x RTX 3090, 24 GiB each; driver 570.144.
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

A forced-token test generated `[151666, 151666, 151666, 151666]`, but it does not count as latent-runtime verification because the intended spawn-safe patch file was absent from the server. Therefore it only proves forced token sampling, not Monet hidden-state substitution.

## Official Patch Clarification
The pinned Monet README gives a VLMEvalKit patch that copies `monet_gpu_model_runner.py` into a `Monet_models` directory and injects it as `vllm.v1.worker.gpu_model_runner` from a startup hook. The README command names the file `sitecustomized.py`, while its own comment says `sitecustomize.py`; Python's automatic startup hook uses `sitecustomize.py`.

To keep baseline reproduction clean, the next check uses the official README patch body with only this filename correction. It does not use the earlier V_COT-enhanced `sitecustomize.py` implementation.

## Current Task
Run `scripts/06_official_runner_check.sh`.

This lightweight script:
- verifies the pinned Monet commit;
- creates a temporary `Monet_models` directory;
- copies the official pinned `inference/vllm/monet_gpu_model_runner.py` into it;
- creates `sitecustomize.py` using the official README patch body, changing only the filename from the README typo;
- starts both a parent Python process and a `spawn` child process;
- verifies that both resolve `vllm.v1.worker.gpu_model_runner` to the copied Monet runner;
- loads no model weights and performs no benchmark inference.

Success requires:
- `PARENT_runner_file=.../Monet_models/monet_gpu_model_runner.py`;
- `CHILD_runner_file=.../Monet_models/monet_gpu_model_runner.py`;
- `CHILD_exitcode=0`;
- `OFFICIAL_RUNNER_PATCH_PASS=True`.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [ ] Verify official-style runner patch in both parent and spawned child.
- [ ] Exercise the latent hidden-state path with a deterministic forced-start diagnostic using the verified official patch path.
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
Transfer and run `scripts/06_official_runner_check.sh`. Return the `PARENT_runner_file`, `CHILD_runner_file`, `CHILD_exitcode`, and `OFFICIAL_RUNNER_PATCH_PASS` lines. Do not reload Monet-7B until this lightweight spawned-runner check passes.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
