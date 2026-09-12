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
- Protected Monet runtime remains: `torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`, `vllm==0.10.0`, `transformers==4.54.0`, `trl==0.15.2`.
- Additional protected Hugging Face packages verified after VLMEvalKit dependency bring-up: `huggingface-hub==0.36.2`, `tokenizers==0.21.4`, `accelerate==1.15.0`, `datasets==5.0.1`, `qwen-vl-utils==0.0.14`.
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

## VLMEvalKit Dependency Bring-Up — COMPLETE
A full unconstrained `pip install -r requirements.txt` was intentionally avoided because a resolver dry-run would upgrade `transformers` from verified `4.54.0` to `5.17.0`, move the Hugging Face stack, install duplicate OpenCV / dotenv distributions, and downgrade `pylatexenc`.

Controlled bring-up instead used a filtered dependency set and constraints freezing the verified Monet/Hugging Face stack. A source-level import audit identified `rouge` as the only undeclared hard external import. `rouge==1.0.1` was added with `--no-deps`. `setuptools` was pinned to `81.0.0` to restore the legacy `pkg_resources` API required by `openai-clip`.

Final import verification on 2026-09-12:
```text
vlmeval import: PASS
vlmeval version: 0.2rc1
```

Non-fatal messages still observed:
- missing `.env` warning from `vlmeval.smp.misc.load_env`;
- `pkg_resources` deprecation warning emitted by Jieba;
- `TRANSFORMERS_CACHE` deprecation warning emitted by Transformers;
- `pip check` may still report `decord 0.6.0 is not supported on this platform`; `decord` imports successfully and this is treated as a video-only compatibility warning, not a blocker for image-only VStarBench.

Dependency bring-up is closed. Do not install additional optional packages unless a selected evaluation path demonstrably requires them.

## Official Monet Evaluation Integration Facts
Pinned Monet README specifies the following for VLMEvalKit evaluation:
- use `vllm==0.10.0`;
- copy `Monet/inference/vllm/monet_gpu_model_runner.py` into `VLMEvalKit/Monet_models/`;
- use a Python startup hook to replace `vllm.v1.worker.gpu_model_runner` with the Monet runner and set latent IDs `151666/151667`;
- use the system prompt: `You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}.`;
- use an API model as supplementary judge for exact reproduction of reported benchmark scores.

The Monet README names the startup file `sitecustomized.py`, but Python startup autoload requires `sitecustomize.py`; V_COT already verified the official patch body using the correct filename in both parent and spawned worker processes.

## Pinned VLMEvalKit Qwen2.5-VL / VStar Path — INSPECTED
Exact source inspection at the selected VLMEvalKit snapshot established:

### Model wrapper and vLLM construction
`vlmeval/vlm/qwen2_vl/model.py` uses `Qwen2VLChat`. With `use_vllm=True`, the wrapper creates `vllm.LLM` with:
- `max_num_seqs=5`;
- `max_model_len=32768`;
- image limit 24 per prompt;
- GPU memory utilization default 0.9;
- automatic tensor-parallel size according to visible GPU count.

This differs from Monet's standalone helper setting (`max_model_len=4096`). For VLMEvalKit-based reproduction, do not silently replace the wrapper's `32768`; record the difference and preserve the evaluation path unless an upstream Monet instruction says otherwise.

### Actual vLLM sampling
Although the wrapper constructor exposes sampling defaults, `generate_inner_vllm` explicitly creates:
```python
SamplingParams(
    temperature=0.0,
    max_tokens=self.max_new_tokens,
    stop_token_ids=None,
)
```
Thus the vLLM evaluation path is greedy (`temperature=0.0`) with the wrapper's `max_new_tokens` limit (default 2048).

### Raw token IDs are currently discarded
After `self.llm.generate(...)`, pinned VLMEvalKit does:
```python
for o in outputs:
    generated_text = o.outputs[0].text
...
return generated_text
```
Therefore normal VLMEvalKit output retains only generated text and discards `o.outputs[0].token_ids`. This is the exact point that must be instrumented for natural latent-trigger characterization. Capturing `o.outputs[0].token_ids` at this point can be observational only and need not change sampling or model behavior.

### VStarBench prompt path
`VStarBench` is registered in `ImageMCQDataset`, whose `TYPE = 'MCQ'`. Its standard dataset prompt is image(s) followed by text containing optional hint, question, options, and:
`Please select the correct answer from the options above.`

The pinned `Qwen2.5-VL-7B-Instruct` model registration sets `use_custom_prompt=False`, so the dataset's `ImageMCQDataset.build_prompt` path is the relevant baseline prompt behavior rather than Qwen2VLPromptMixin's custom MCQ path.

The Monet system prompt is independently inserted by `Qwen2VLChat` when `system_prompt` is supplied, before the user content, and therefore can coexist with the standard VStarBench dataset prompt exactly as Monet requests.

### Custom config path
Pinned `run.py` supports JSON model configs containing `class` plus class-specific constructor arguments. This allows a Monet evaluation entry to use `class: Qwen2VLChat`, the local Monet checkpoint path, `use_vllm: true`, Monet's system prompt, and the desired image preprocessing settings without editing the upstream VLMEvalKit model registry in place.

## Natural Latent-Trigger Status
Natural latent activation is not yet characterized. Planned first scan:
- dataset: `VStarBench`;
- initial subset: 20–50 examples;
- `LATENT_SIZE=10`;
- no forced tokens and no `allowed_token_ids`;
- generation semantics identical to the pinned VLMEvalKit vLLM path;
- record raw token IDs, raw text, start-token presence (`151666`), end-token presence (`151667`), and latent-segment count;
- compute trigger rate only after the raw-token path is verified.

Before loading Monet for that scan, first perform a lightweight dataset/plumbing check: build `VStarBench`, verify download/MD5/image extraction, and print the first standard prompt. This isolates dataset/network/prompt issues from model inference.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [x] Verify official-style runner patch in both parent and spawned child.
- [x] Exercise the official Monet latent runtime path with a deterministic forced diagnostic.
- [x] Place a reproducible VLMEvalKit snapshot and verify VStarBench is present.
- [x] Complete controlled VLMEvalKit dependency bring-up.
- [x] Inspect exact Monet/VLMEvalKit evaluation integration and raw token-ID capture point.
- [ ] Verify VStarBench dataset build and first prompt without model loading.
- [ ] Implement observational raw-token capture without changing generation semantics.
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
Do not install anything else and do not load Monet yet. Build the pinned `VStarBench` dataset through VLMEvalKit and inspect the first sample's standard prompt/image path. If this passes, create the observational raw-token capture path at `o.outputs[0].token_ids` and then run the first natural latent-trigger scan with `LATENT_SIZE=10` and no forced tokens.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
