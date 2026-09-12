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

## VLMEvalKit Dependency Audit
A full `pip install -e .` / unconstrained `pip install -r requirements.txt` remains prohibited because the pinned VLMEvalKit requirements are broad and mostly unpinned.

Verified protected Monet stack remains unchanged:
- `torch==2.7.1`
- `torchvision==0.22.1`
- `transformers==4.54.0`
- `trl==0.15.2`
- `vllm==0.10.0`

Incrementally added without a full VLMEvalKit install:
- `validators==0.35.0`
- `matplotlib==3.10.9` plus required runtime dependencies
- `tabulate==0.10.0`
- `sty==1.0.6`
- `portalocker`
- `Levenshtein==0.27.1`
- `RapidFuzz==3.14.5`
- `imageio`
- `decord==0.6.0`
- `timeout-decorator`
- `jieba==0.42.1`

Read-only direct audit of the pinned VLMEvalKit `requirements.txt`:
- 32 direct requirements currently satisfied.
- 29 direct requirements still missing: `anls`, `antlr4-python3-runtime`, `apted`, `bert_score`, `cairosvg`, `colormath`, `distance`, `editdistance`, `google-genai`, `gradio`, `ipdb`, `json_repair`, `lpips`, `lxml`, `math-verify`, `nltk`, `num2words`, `omegaconf`, `openai-clip`, `openpyxl`, `pdf2image`, `polygon3`, `scikit-image`, `scikit-learn`, `sentence_transformers`, `timm`, `torchmetrics`, `xlsxwriter`, and `zss`.
- One direct version mismatch: pinned requirement `pylatexenc==2.10`, installed `pylatexenc==2.11`.
- `dotenv` is functionally provided by `python-dotenv==1.2.3`; do not install the separate `dotenv` distribution unless demonstrated necessary.
- `opencv-python>=4.7.0.72` is functionally provided by `opencv-python-headless==5.0.0.93` / importable `cv2`; do not install a second OpenCV distribution.
- `pip check` reports `decord 0.6.0 is not supported on this platform`; `import decord` itself succeeded sufficiently for VLMEvalKit to progress beyond that top-level import, and VStarBench is image-only. Keep this as a separate video-only issue.

## Resolver Dry-Run Results
An unconstrained dry-run of the full VLMEvalKit requirements planned 76 package actions and would upgrade `transformers` from verified `4.54.0` to `5.17.0`, move `huggingface_hub` and `tokenizers`, install a second OpenCV distribution, install the separate `dotenv` distribution, and downgrade `pylatexenc`. That path is unsafe and remains prohibited.

A second dry-run was then performed on only the 29 missing direct non-core dependencies under constraints freezing the verified Monet/Hugging Face stack. This filtered/constrained plan is clean:
- 70 total package installations are planned, including transitive dependencies.
- `PROTECTED PACKAGE ACTIONS = NONE`.
- No action is planned for `torch`, `torchvision`, `transformers`, `trl`, `vllm`, `huggingface_hub`, `tokenizers`, `accelerate`, `datasets`, or `qwen-vl-utils`.
- The resolved direct package versions include `nltk==3.10.3`, `sentence-transformers==5.7.0`, `gradio==6.17.3`, `timm==1.0.29`, `torchmetrics==1.9.0`, `scikit-learn==1.7.2`, and the other audited missing packages.

Therefore a real batch installation of the exact filtered requirements under the same constraints is now acceptable. Preserve a pre-install environment snapshot first and verify the protected stack again afterward.

## Dependency Strategy
Use a filtered, constrained batch installation instead of either full requirements installation or serial one-by-one blocker chasing:
1. Use `/tmp/vlmeval_missing_direct.txt` containing only the 29 missing direct non-core dependencies.
2. Use `/tmp/vcot_constraints.txt` to freeze the verified Monet/Hugging Face stack.
3. Save a pre-install `pip freeze` snapshot for rollback/reference.
4. Install the filtered set with normal dependency resolution under the constraints.
5. Re-check protected package versions, run `pip check`, and retry `import vlmeval` with full traceback capture.
6. Keep `pylatexenc==2.11`, the existing headless OpenCV provider, and the current `decord` warning unchanged unless they become a demonstrated blocker.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [x] Verify official-style runner patch in both parent and spawned child.
- [x] Exercise the official Monet latent hidden-state path with deterministic forced-start/end behavior.
- [x] Place a reproducible VLMEvalKit snapshot and verify VStarBench is present.
- [x] Complete read-only direct dependency audit for the pinned VLMEvalKit snapshot.
- [x] Inspect the full resolver dry-run and confirm that an unconstrained install would modify protected Monet dependencies.
- [x] Run filtered/constrained resolver dry-run for the 29 missing non-core dependencies and verify `PROTECTED PACKAGE ACTIONS = NONE`.
- [ ] Complete controlled batch VLMEvalKit dependency bring-up.
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
- `decord==0.6.0` currently triggers a `pip check` platform-support warning even though its Python import succeeds; do not treat `pip check` cleanliness as the sole gate for VStarBench.
- vLLM shutdown may emit NCCL/resource-tracker warnings after successful inference; treat them as cleanup warnings unless they cause reproducible resource accumulation.
- Forced latent-token diagnostics are engineering tests only and must never be mixed with benchmark results.

## Next Action
Save a pre-install `pip freeze` snapshot, install the 29 filtered direct dependencies with normal transitive resolution under `/tmp/vcot_constraints.txt`, then verify the protected stack, run `pip check`, and retry `import vlmeval` with full traceback capture. Do not install the full VLMEvalKit requirements file directly.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.