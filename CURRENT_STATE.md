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
- Verified runtime stack before VLMEvalKit dependency bring-up: `torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`, `vllm==0.10.0`, `transformers==4.54.0`, `trl==0.15.2`.
- CUDA tensor test passed.
- Monet-7B checkpoint is fully downloaded and structurally verified.
- The pinned VLMEvalKit snapshot is extracted at `~/work/V_COT/third_party/VLMEvalKit` and contains the `VStarBench` dataset entry.

## Verified Baseline Inference
- Official Monet example runs successfully and returns `\\boxed{C}`.
- `<abs_vis_token>` -> `151666`.
- `</abs_vis_token>` -> `151667`.
- `scripts/06_official_runner_check.sh` passed in parent and spawned child processes.
- `scripts/07_official_forced_latent_path.sh` passed; deterministic forced latent IDs matched `[151666, 151666, 151667]` with `LATENT_SIZE=2`.
- Forced-token checks are engineering diagnostics only, not benchmark evidence.

## Natural Latent-Trigger Status
Natural latent activation is still not characterized. The next scientific baseline step remains a small natural-trigger scan on `VStarBench` without forced tokens after evaluation plumbing is ready.

## VLMEvalKit Dependency Bring-Up
A full unconstrained `pip install -r requirements.txt` remains prohibited. A resolver dry-run showed that it would upgrade `transformers` from verified `4.54.0` to `5.17.0`, move the Hugging Face stack, install duplicate OpenCV / dotenv distributions, and downgrade `pylatexenc`.

A filtered audit identified 29 missing direct non-core requirements. A constrained dry-run for those packages planned 70 installs including transitive dependencies and reported `PROTECTED PACKAGE ACTIONS = NONE`. The filtered batch was then installed under the protected constraints.

After the controlled batch:
- `pip check` reports only `decord 0.6.0 is not supported on this platform`.
- This `decord` warning remains treated as a separate video-only compatibility issue because its import had already succeeded and the current target VStarBench is image-only.
- `import vlmeval` now advances beyond the previously missing packages but fails at `vlmeval/dataset/foxbench.py` with `ModuleNotFoundError: No module named 'rouge'`.
- `foxbench.py` contains a top-level `from rouge import Rouge` import.
- The pinned VLMEvalKit `requirements.txt` does not list `rouge`, so the previous direct-requirement audit could not discover this blocker.
- The `.env` warning emitted by `load_env` remains non-fatal.

## Dependency Strategy
The direct-requirement audit is no longer sufficient because the repository contains import-time dependencies that are not declared in `requirements.txt`. Do not return to blind one-by-one installation. Instead:
1. Keep the protected Monet/Hugging Face stack fixed.
2. Perform a read-only static source-import audit starting from the pinned local `vlmeval` package to identify missing external import modules reachable from top-level package imports.
3. Verify the protected stack again after the controlled batch installation.
4. Batch-handle only the missing undeclared imports that are actually relevant to import-time package initialization.
5. Retry `import vlmeval` only after this source-level audit.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [x] Verify official-style runner patch in both parent and spawned child.
- [x] Exercise the official Monet latent runtime path with a deterministic forced diagnostic.
- [x] Place a reproducible VLMEvalKit snapshot and verify VStarBench is present.
- [x] Complete direct dependency audit for the pinned VLMEvalKit snapshot.
- [x] Confirm unconstrained installation would modify protected Monet dependencies.
- [x] Run filtered/constrained dry-run and verify `PROTECTED PACKAGE ACTIONS = NONE`.
- [x] Install the filtered missing direct dependencies under constraints.
- [ ] Complete source-level audit for undeclared import-time dependencies.
- [ ] Complete controlled VLMEvalKit import bring-up.
- [ ] Measure natural latent-trigger frequency on a VStarBench subset.
- [ ] Reproduce selected Monet benchmark baseline under documented settings.
- [ ] Freeze reproduced baseline with a Git tag.
- [ ] Locate and instrument exact latent-state tensors needed for V0 experiments.
- [ ] Start V0 only after the above checks pass.

## Known Issues
- GitHub access from the GPU server is unavailable; source synchronization must use local staging or direct file handoff.
- The VLMEvalKit snapshot was transferred as an archive and has no `.git`; the selected snapshot commit is documented here instead.
- Windows-to-Linux transfers may convert LF to CRLF.
- `huggingface_hub` HEAD metadata calls are incompatible with the current HF mirror for the Monet checkpoint; direct resumable GET is the verified workaround.
- `nvcc` is not installed system-wide.
- `decord==0.6.0` triggers a platform-support warning in `pip check` even though Python import succeeds; do not use `pip check` cleanliness as the sole VStarBench gate.
- vLLM shutdown may emit NCCL/resource-tracker cleanup warnings after successful inference.

## Next Action
Run a read-only source-level import audit on the local pinned VLMEvalKit tree and report all missing external modules reachable from package top-level imports, together with the protected Monet/Hugging Face package versions. Do not install `rouge` yet; first determine whether there are additional undeclared import-time dependencies so they can be handled in one controlled batch.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.