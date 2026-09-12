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
The official Monet example runs successfully and returns the correct answer (`\\boxed{C}`). Model loading, multimodal preprocessing, and generation complete successfully under the pinned runtime.

This marks **official-example inference as reproduced**.

## Latent-Mode Status
Two generation runs have not yet shown a natural latent trigger:
1. the unmodified official example did not emit `<abs_vis_token>`;
2. a diagnostic prompt explicitly asking the model to begin with `<abs_vis_token>` also did not make the model emit it.

The tokenizer wiring is verified:
- `<abs_vis_token>` -> `151666`
- `</abs_vis_token>` -> `151667`
- `LATENT_SIZE=10`

Therefore the remaining question is not token registration. We need to separate:
- **natural trigger behavior**: whether the checkpoint chooses token 151666 on its own;
- **runner-path correctness**: whether the customized vLLM worker enters the hidden-state latent path after token 151666 is sampled.

A second implementation concern is now explicit: vLLM 0.10.0 uses a spawned EngineCore process. Patching `sys.modules` only in the parent interpreter is not sufficient evidence that the spawned worker also uses Monet's `GPUModelRunner`. Monet's README describes patching every spawned process for evaluation; the next diagnostic therefore uses Python's standard `sitecustomize.py` mechanism so the patch is applied at interpreter startup in parent and spawned worker processes.

Monet's inference runner code confirms the intended latent mechanics: after a sampled token equals `LATENT_START_ID`, it sets latent state active and stores the current last-layer hidden state as `pending`; on the next decode step, when active and pending is present, that hidden vector overwrites the token embedding for the request. The state exits on `LATENT_END_ID` or after `LATENT_SIZE` steps.

## Current Task
Run a deterministic engineering-only latent-path test using:
- `scripts/monet_site/sitecustomize.py`: spawn-safe Monet runner patch;
- `scripts/06_force_latent_path.py`;
- `scripts/06_force_latent_path.sh`.

The test uses vLLM V1's built-in `allowed_token_ids=[151666]` for a very short four-token diagnostic request. This deliberately forces sampling of the latent-start token. It is **not** a benchmark configuration and cannot be used as scientific evidence of natural latent triggering.

Success criteria:
1. log contains `[VCOT_MONET_SITE]` from the parent and spawned worker path;
2. Monet runner reports `start=151666`, `end=151667`, `latent_size=10` (or equivalent initialization evidence);
3. generated token IDs begin with `151666` and contain at least two decode steps;
4. no runner/runtime error occurs during subsequent decode steps.

If these pass, the hidden-state latent code path is operational even though natural checkpoint triggering is still unverified.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [ ] Verify the spawned worker is actually using the Monet custom runner.
- [ ] Exercise the latent hidden-state path with a deterministic forced-start diagnostic.
- [ ] Observe at least one natural latent-mode activation from the checkpoint, or document its trigger rate on an appropriate benchmark subset.
- [ ] Reproduce selected Monet benchmark baseline under documented settings.
- [ ] Freeze reproduced baseline with a Git tag.
- [ ] Locate and instrument the exact latent-state tensors needed for V0 experiments.
- [ ] Start V0 only after the above checks pass.

## Hardware Plan
- Development/debug/V0 pilot: 4 x RTX 3090 when needed.
- Lightweight smoke/diagnostic inference: one currently free RTX 3090.
- Full-scale or RL/VLPO experiments may later use more 3090s or H200 if justified by memory/runtime.

## Active Method Version
None. V0 has not started.

## Current Experiment
Baseline reproduction / latent-runtime diagnostics only; no scientific-method experiment has started.

## Known Issues
- GitHub access from the GPU server is unavailable; source synchronization must use local staging or direct file handoff.
- `huggingface_hub` HEAD metadata calls are incompatible with the current HF mirror for this checkpoint; direct resumable GET is the verified workaround.
- Windows-to-Linux transfers may convert LF to CRLF; normalize transferred shell scripts before execution.
- `nvcc` is not installed system-wide. This is not a blocker for inference but may matter later for training extensions.
- vLLM shutdown may emit NCCL/resource-tracker warnings after successful inference; treat them as cleanup warnings unless they cause reproducible resource accumulation.
- The absence of `<abs_vis_token>` in one or two examples must not be interpreted as evidence that Monet lacks latent reasoning; natural trigger frequency has not yet been measured.
- Forced latent-token diagnostics are engineering tests only and must never be mixed with baseline benchmark results.

## Next Action
Transfer and run the three step-06 diagnostic files. Return the `[VCOT_MONET_SITE]` lines, any `start_id/end_id/latent_size` runner initialization line, `RAW OUTPUT`, `TOKEN IDS`, and `FORCED PATH CHECK`. Do not start benchmark evaluation or V0 development until this runner-path check is resolved.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
