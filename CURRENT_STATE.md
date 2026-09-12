# Current State

## Project Stage
Baseline reproduction / natural latent-trigger characterization.

## Current Objective
Establish a reproducible Monet baseline before implementing any new latent-supervision method.

## Current Branch
`main`

## Selected Upstream Baseline
- Monet repository: `https://github.com/NOVAglow646/Monet.git`
- Pinned Monet commit: `08939998d3d643a73a316e349faa34f420429153`
- Primary checkpoint: `NOVAglow646/Monet-7B`
- Baseline specification: `BASELINE.md`
- VLMEvalKit snapshot selected for reproducibility: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3` (chosen by V_COT; Monet does not pin a VLMEvalKit commit in its README).

## Verified Infrastructure
Verified on 2026-09-12:
- Monet source is present at `~/work/V_COT/third_party/Monet` and HEAD is exactly `08939998d3d643a73a316e349faa34f420429153`.
- `vcot` Conda environment exists with Python 3.10.21.
- Server has 10 x RTX 3090, 24 GiB each; driver 570.144.
- Verified runtime stack: `torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`, `vllm==0.10.0`, `transformers==4.54.0`, `trl==0.15.2`.
- CUDA is available and a CUDA tensor test passed.
- Monet-7B checkpoint is fully downloaded and structurally verified; four safetensors shards total 15.44 GiB, matching the checkpoint index.
- The pinned VLMEvalKit snapshot is extracted at `~/work/V_COT/third_party/VLMEvalKit` and contains the `VStarBench` dataset entry.

## Verified Baseline Inference
The official Monet example runs successfully and returns the correct answer (`\\boxed{C}`).

## Verified Token Wiring
- `<abs_vis_token>` -> `151666`
- `</abs_vis_token>` -> `151667`
- standard inference setting tested with `LATENT_SIZE=10`.

## Verified Official-Style Runner Patch
`scripts/06_official_runner_check.sh` passed on 2026-09-12 in both parent and spawned child processes.

## Verified Official Monet Latent Runtime Path
`scripts/07_official_forced_latent_path.sh` passed on 2026-09-12.
Observed token IDs exactly matched `[151666, 151666, 151667]` with `LATENT_SIZE=2`, confirming the official Monet latent state machine is operational under the pinned runtime. This remains an engineering-only diagnostic, not benchmark evidence.

## Natural Latent-Trigger Status
Natural latent activation is still not characterized. The next scientific baseline step is a small natural-trigger scan on `VStarBench` without forced tokens.

## VLMEvalKit Dependency Bring-Up
A full `pip install -e .` remains intentionally avoided because VLMEvalKit's requirements include broad, mostly unpinned dependencies including `torch`, `torchvision`, and `transformers`, plus many dependencies for unrelated benchmarks.

Verified dependency state so far:
- Monet-critical packages remain intact.
- `validators==0.35.0`, `matplotlib==3.10.9` plus its runtime dependencies, `tabulate==0.10.0`, `sty==1.0.6`, `portalocker`, `Levenshtein==0.27.1`, `RapidFuzz==3.14.5`, `imageio`, `decord==0.6.0`, `timeout-decorator`, and `jieba==0.42.1` have been added incrementally without a full VLMEvalKit install.
- `pip check` has remained clean after each resolved dependency layer unless explicitly noted and fixed.
- The `.env` message emitted by `load_env` is non-fatal and is not the import blocker.
- After resolving `jieba`, the next observed import blocker is `ModuleNotFoundError: No module named 'nltk'` from `vlmeval/dataset/foxbench.py`.

The one-by-one strategy was chosen initially to protect the already verified Monet runtime and to avoid installing a large set of benchmark-specific packages that VStarBench itself may not need. However, VLMEvalKit's broad top-level import chain is causing many unrelated benchmark dependencies to appear serially. The strategy is therefore changed: perform a read-only batch audit of the entire pinned `requirements.txt` against the current `vcot` environment, including version-specifier checks, before installing anything else. Then install only the audited missing non-core dependencies, while keeping Monet-critical versions fixed.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [x] Verify official-style runner patch in both parent and spawned child.
- [x] Exercise the official Monet latent hidden-state path with deterministic forced-start/end behavior.
- [x] Place a reproducible VLMEvalKit snapshot and verify VStarBench is present.
- [ ] Complete batch dependency audit for the pinned VLMEvalKit snapshot without mutating the environment.
- [ ] Complete minimum VLMEvalKit dependency bring-up without modifying Monet-critical versions.
- [ ] Measure natural latent-trigger frequency on a VStarBench subset.
- [ ] Reproduce selected Monet benchmark baseline under documented settings.
- [ ] Freeze reproduced baseline with a Git tag.
- [ ] Locate and instrument the exact latent-state tensors needed for V0 experiments.
- [ ] Start V0 only after the above checks pass.

## Known Issues
- GitHub access from the GPU server is unavailable; source synchronization must use local staging or direct file handoff.
- The VLMEvalKit snapshot was transferred as an archive, so its directory does not contain `.git`; the selected snapshot commit is documented in V_COT instead.
- Windows-to-Linux transfers may convert LF to CRLF; normalize transferred shell scripts before execution.
- `huggingface_hub` HEAD metadata calls are incompatible with the current HF mirror for the Monet checkpoint; direct resumable GET is the verified workaround.
- `nvcc` is not installed system-wide.
- vLLM shutdown may emit NCCL/resource-tracker warnings after successful inference; treat them as cleanup warnings unless they cause reproducible resource accumulation.
- Forced latent-token diagnostics are engineering tests only and must never be mixed with benchmark results.

## Next Action
Run a read-only full audit of `third_party/VLMEvalKit/requirements.txt` against the active `vcot` environment. Report installed direct requirements, missing direct requirements, version mismatches, and the protected Monet-critical package versions. Also run `pip check` to expose missing transitive dependencies. Do not install `nltk` or any additional package until the audit result is inspected.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
