# Current State

## Project Stage
Baseline reproduction / natural latent-trigger characterization.

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

## Verified Token Wiring
- `<abs_vis_token>` -> `151666`
- `</abs_vis_token>` -> `151667`
- standard inference setting tested with `LATENT_SIZE=10`.

## Verified Official-Style Runner Patch
`scripts/06_official_runner_check.sh` passed on 2026-09-12.

Using the Monet README patch body with only the `sitecustomized.py` -> `sitecustomize.py` filename correction, both the parent process and a Python `spawn` child resolved the vLLM runner to the copied official Monet runner:
- `PARENT_runner_file=.../Monet_models/monet_gpu_model_runner.py`
- `CHILD_runner_file=.../Monet_models/monet_gpu_model_runner.py`
- `CHILD_exitcode=0`
- `OFFICIAL_RUNNER_PATCH_PASS=True`

This closes the spawn-patching uncertainty.

## Verified Official Monet Latent Runtime Path
`scripts/07_official_forced_latent_path.sh` passed on 2026-09-12.

Engineering-only diagnostic settings:
- `LATENT_START_ID=151666`
- `LATENT_END_ID=151667`
- `LATENT_SIZE=2`
- sampler constrained with `allowed_token_ids=[151666]`
- `max_tokens=3`

Observed runner initialization:
- `start_id=151666 end_id=151667, latent_size=2`
- spawned processes printed `Replaced the original vllm gpu_model_runner with the Monet version.`

Observed token sequence:
- expected: `[151666, 151666, 151667]`
- observed: `[151666, 151666, 151667]`
- `OFFICIAL_LATENT_PATH_PASS=True`

This verifies that the official Monet latent state machine is operational under the pinned runtime. The forced end token demonstrates that the custom runner entered active latent state and applied its latent-length termination logic; because the verified official runner uses the pending last-layer hidden state to overwrite the next-step input embedding while active, the hidden-state latent path is exercised between the latent-start and latent-end steps.

This forced-token diagnostic is an engineering verification only and must not be used as benchmark evidence or evidence of natural trigger frequency.

## Natural Latent-Trigger Status
Natural latent activation is still not characterized:
1. the unmodified official example did not emit `<abs_vis_token>`;
2. explicitly asking the model to begin with `<abs_vis_token>` also did not make it emit the token.

These two examples are insufficient to infer the checkpoint's natural trigger frequency.

The Monet paper evaluates Monet-7B on V*, HRBench4K, HRBench8K, MME-RealWorld-Lite, and VisualPuzzles using VLMEvalKit. The next baseline step is to measure natural `<abs_vis_token>` activation on a small real benchmark subset before full benchmark reproduction. V* is the preferred first subset because it is an official Monet benchmark and directly tests fine-grained visual perception/search where the paper reports gains from latent reasoning.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [x] Verify official-style runner patch in both parent and spawned child.
- [x] Exercise the official Monet latent hidden-state path with deterministic forced-start/end behavior.
- [ ] Measure natural latent-trigger frequency on an official benchmark subset.
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
Prepare the official VLMEvalKit/V* evaluation path and run a small natural-trigger scan without forced tokens. Record, per sample, whether raw output contains `<abs_vis_token>` and `</abs_vis_token>`, then report trigger count/rate. Do not start V0 yet.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
