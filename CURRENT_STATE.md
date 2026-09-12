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
- standard inference setting tested with `LATENT_SIZE=10`.

A previous forced-token test generated `[151666, 151666, 151666, 151666]`, but that run did not use the intended startup patch and therefore is not counted as latent-runtime verification.

## Verified Official-Style Runner Patch
`scripts/06_official_runner_check.sh` passed on 2026-09-12.

Using the Monet README patch body with only the `sitecustomized.py` -> `sitecustomize.py` filename correction, both the parent process and a Python `spawn` child resolved the vLLM runner to the copied official Monet runner:
- `PARENT_runner_file=.../Monet_models/monet_gpu_model_runner.py`
- `PARENT_runner_class_module=Monet_models.monet_gpu_model_runner`
- `CHILD_runner_file=.../Monet_models/monet_gpu_model_runner.py`
- `CHILD_runner_class_module=Monet_models.monet_gpu_model_runner`
- `CHILD_exitcode=0`
- `OFFICIAL_RUNNER_PATCH_PASS=True`

The log also printed `Replaced the original vllm gpu_model_runner with the Monet version.` in both parent/spawned-process startup paths. This closes the spawn-patching uncertainty: the official Monet GPUModelRunner can be loaded by the actual spawned Python process.

## Current Task
Run `scripts/07_official_forced_latent_path.sh`.

This is an engineering-only latent state-machine test using the verified official-style patch path. It does not modify the Monet runner.

Diagnostic settings:
- `LATENT_START_ID=151666`
- `LATENT_END_ID=151667`
- `LATENT_SIZE=2` only for this short diagnostic
- vLLM sampler is constrained with `allowed_token_ids=[151666]`
- `max_tokens=3`

Expected behavior if the official Monet latent state machine is active:
1. token 1 samples `151666`, activating latent state;
2. token 2 samples `151666` while the runner is active and uses the pending last-layer hidden state as the next-step input embedding;
3. on token 3, because the latent length has reached 2, the Monet runner rewrites the sampled token to `151667`.

Therefore the expected visible token sequence is exactly:
`[151666, 151666, 151667]`.

A plain vLLM runner under the same allowed-token constraint would remain `[151666, 151666, 151666]`. Thus observing the forced end token provides visible evidence that the Monet latent state machine executed, while the already verified runner code path implies the hidden-state embedding substitution is exercised between the first and second latent steps.

This diagnostic is not a benchmark setting and must not be used as scientific evidence of natural latent triggering.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [x] Verify official-style runner patch in both parent and spawned child.
- [ ] Exercise the official Monet latent hidden-state path with deterministic forced-start/end behavior.
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
Transfer and run `scripts/07_official_forced_latent_path.sh`. Return the `RUNNER CHECK`, `TOKEN IDS`, `EXPECTATION`, and `OFFICIAL_LATENT_PATH_PASS` sections. Do not begin benchmark evaluation or V0 development until this deterministic official latent-path test passes.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
