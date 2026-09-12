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
- This `decord` warning remains a separate video-only compatibility issue because its import succeeds and the target VStarBench is image-only.
- `import vlmeval` advances to `vlmeval/dataset/foxbench.py` and fails on undeclared `rouge`.
- The pinned VLMEvalKit `requirements.txt` does not list `rouge`.
- The `.env` warning emitted by `load_env` remains non-fatal.

## Source-Level Import Audit
A read-only static import-graph audit was completed starting from top-level `vlmeval` import.

Verified protected stack after the controlled batch:
- `torch==2.7.1`
- `torchvision==0.22.1`
- `transformers==4.54.0`
- `trl==0.15.2`
- `vllm==0.10.0`
- `huggingface-hub==0.36.2`
- `tokenizers==0.21.4`
- `accelerate==1.15.0`
- `datasets==5.0.1`
- `qwen-vl-utils==0.0.14`

Audit results:
- 402 local modules are reachable from top-level `vlmeval` import.
- Exactly one unguarded external module is missing: `rouge`.
- Blocking source location: `vlmeval/dataset/foxbench.py:12 -> from rouge import Rouge`.
- Optional/guarded-only missing modules: `anthropic`, `boto3`, `botocore`, `flash_attn`, and `vertexai`.
- Those five optional modules do not block top-level `import vlmeval` and should not be installed for the VStarBench path unless later required.
- No Python parse errors were found in the audit.
- PyPI package `rouge==1.0.1` provides the exact `from rouge import Rouge` API used by `foxbench.py`.

## Dependency Strategy
1. Keep the protected Monet/Hugging Face stack fixed.
2. Do not install optional cloud/video/FlashAttention dependencies unless they become relevant.
3. Install only `rouge==1.0.1` with `--no-deps`; its purpose is solely to satisfy the remaining hard import blocker.
4. Re-run protected-stack verification, `pip check`, and `import vlmeval`.
5. If `import vlmeval` passes, stop dependency bring-up and move directly to inspecting the exact Monet/VLMEvalKit evaluation integration and raw token-ID capture path for VStarBench.

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
- [x] Complete source-level audit for undeclared import-time dependencies.
- [ ] Install `rouge==1.0.1` and verify `import vlmeval` passes.
- [ ] Inspect exact Monet/VLMEvalKit evaluation integration and raw token-ID capture path.
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
Install only `rouge==1.0.1` with `--no-deps`, then verify the protected stack remains unchanged, run `pip check`, and retry `import vlmeval` with full traceback capture. Do not install the five optional guarded modules.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.