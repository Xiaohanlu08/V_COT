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
- TUNA PyPI mirror and `https://hf-mirror.net` are reachable from the GPU server; direct GitHub access is not reliable.
- `vcot` Conda environment exists with Python 3.10.21.
- Server has 10 x RTX 3090, 24 GiB each; driver 570.144; `nvidia-smi` reports CUDA 12.8 capability.
- Verified runtime stack:
  - `torch==2.7.1+cu126`
  - `torchvision==0.22.1+cu126`
  - `vllm==0.10.0`
  - `transformers==4.54.0`
  - `trl==0.15.2`
- CUDA is available and a CUDA tensor test passed.
- Monet-7B checkpoint is fully downloaded and structurally verified.
- Four safetensors shards total exactly 15.44 GiB, matching `model.safetensors.index.json`.

## Verified Baseline Inference
The official Monet example now runs successfully through the customized vLLM path.

Observed output:
- model returned a coherent non-empty answer;
- predicted `\\boxed{C}`, matching the image evidence in the official example;
- no runtime/CUDA/vLLM initialization error occurred;
- a shutdown warning about one leaked semaphore was observed, but the inference itself completed successfully.

This marks **official-example inference as reproduced**.

## Latent-Mode Status
The successful official-example run did **not** emit `<abs_vis_token>` or `</abs_vis_token>`:
- `contains <abs_vis_token>: False`
- `contains </abs_vis_token>: False`
- `LATENT_SIZE: 10`

This does **not** by itself indicate that the Monet latent runner is broken. The official README states that the model *may* emit `<abs_vis_token>` to enter latent mode; the inference runner activates latent state only when the sampled token ID equals the latent-start ID. Therefore the official example has verified ordinary generation through the Monet-patched runtime, but **natural latent activation is not yet verified**.

The runner code confirms that latent mode is activated only after the sampled token equals `LATENT_START_ID`; once active, it exits on `LATENT_END_ID` or after `LATENT_SIZE` steps.

## Current Task
Run a dedicated latent-mode diagnostic rather than modifying package versions or jumping directly to benchmark evaluation.

New diagnostic files:
- `scripts/05_verify_latent_mode.py`
- `scripts/05_verify_latent_mode.sh`

The diagnostic performs two checks:
1. verify the checkpoint tokenizer maps `<abs_vis_token>` and `</abs_vis_token>` to the expected IDs `151666` and `151667`;
2. run the official image with an explicit diagnostic instruction asking the model to begin with `<abs_vis_token>`, so the vLLM latent path can be exercised if the model follows the learned trigger.

This diagnostic is infrastructure verification only. Its forced instruction must never be used as a benchmark setting or scientific result.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [ ] Verify tokenizer latent special-token IDs.
- [ ] Observe at least one actual latent-mode activation through the customized runner.
- [ ] Reproduce selected Monet benchmark baseline under documented settings.
- [ ] Freeze reproduced baseline with a Git tag.
- [ ] Locate the exact code path that creates/updates continuous latent visual states.
- [ ] Verify latent-state shape, positions, count, and generation behavior.
- [ ] Start V0 only after the above checks pass.

## Hardware Plan
- Development/debug/V0 pilot: 4 x RTX 3090 when needed.
- Lightweight smoke/diagnostic inference: one currently free RTX 3090.
- Full-scale or RL/VLPO experiments may later use more 3090s or H200 if justified by memory/runtime.

## Active Method Version
None. V0 has not started.

## Current Experiment
Baseline reproduction only; no scientific-method experiment has started.

## Known Issues
- GitHub access from the GPU server is unavailable; source synchronization must use local staging or direct file handoff.
- `huggingface_hub` HEAD metadata calls are incompatible with the current HF mirror for this checkpoint; direct resumable GET is the verified workaround.
- Windows-to-Linux transfers may convert LF to CRLF; normalize transferred shell scripts before execution.
- `nvcc` is not installed system-wide. This is not a blocker for inference but may matter later for training extensions.
- `resource_tracker` may report a leaked semaphore at vLLM shutdown after a completed inference; do not treat this warning as a failed inference unless it causes reproducible resource leakage across runs.
- Official Monet SFT scripts target 8 GPUs with DeepSpeed ZeRO-2 and will not be used unmodified as a 4 x RTX 3090 training recipe.

## Next Action
Transfer and run `scripts/05_verify_latent_mode.py` and `scripts/05_verify_latent_mode.sh`. Return the `TOKEN CHECK`, `RAW OUTPUT`, and `LATENT CHECK` sections. Do not begin benchmark evaluation or V0 development until latent-token wiring has been verified.

## Update Rule
After every verified step, update this file with:
- current branch or tag,
- last verified result,
- current task,
- next action,
- known blockers.

This file describes operational state only. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
