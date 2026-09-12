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
- The pinned VLMEvalKit snapshot is extracted at `~/work/V_COT/third_party/VLMEvalKit`.

## Verified Baseline Inference
- Official Monet example runs successfully and returns `\\boxed{C}`.
- `<abs_vis_token>` -> `151666`.
- `</abs_vis_token>` -> `151667`.
- `scripts/06_official_runner_check.sh` passed in parent and spawned child processes.
- `scripts/07_official_forced_latent_path.sh` passed; deterministic forced latent IDs matched `[151666, 151666, 151667]` with `LATENT_SIZE=2`.
- Forced-token checks are engineering diagnostics only, not benchmark evidence.

## VLMEvalKit Dependency Bring-Up — COMPLETE
A full unconstrained `pip install -r requirements.txt` was intentionally avoided because a resolver dry-run would upgrade `transformers` from verified `4.54.0` to `5.17.0` and perturb the Hugging Face stack.

Controlled bring-up instead used a filtered dependency set plus constraints freezing the verified Monet/Hugging Face stack. A source-level import audit identified `rouge` as the only undeclared hard external import. `rouge==1.0.1` was added with `--no-deps`, and `setuptools` was pinned to `81.0.0` to restore legacy `pkg_resources` compatibility required by `openai-clip`.

Final import verification:
```text
vlmeval import: PASS
vlmeval version: 0.2rc1
```

Non-fatal messages that may still appear:
- missing `.env` warning from VLMEvalKit;
- Jieba `pkg_resources` deprecation warning;
- Transformers `TRANSFORMERS_CACHE` deprecation warning;
- `pip check` may still report `decord 0.6.0 is not supported on this platform`; `decord` imports successfully and VStarBench is image-only.

Dependency bring-up is closed. Do not install additional optional packages unless a selected evaluation path demonstrably requires them.

## Official Monet Evaluation Integration Facts
Pinned Monet README specifies the following for VLMEvalKit evaluation:
- use `vllm==0.10.0`;
- copy `Monet/inference/vllm/monet_gpu_model_runner.py` into a `Monet_models` package visible to VLMEvalKit;
- use a Python startup hook to replace `vllm.v1.worker.gpu_model_runner` with the Monet runner and set latent IDs `151666/151667`;
- use the system prompt: `You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}.`;
- use an API model as supplementary judge for exact reproduction of reported benchmark scores.

The Monet README names the startup file `sitecustomized.py`, but Python startup autoload requires `sitecustomize.py`; V_COT already verified the official patch body using the correct filename in both parent and spawned worker processes.

## Pinned VLMEvalKit Qwen2.5-VL / VStar Path — INSPECTED
Exact source inspection at the selected VLMEvalKit snapshot established:
- wrapper: `Qwen2VLChat`;
- vLLM construction uses `max_num_seqs=5`, `max_model_len=32768`, image limit 24, GPU memory utilization 0.9, and automatic tensor-parallel size based on visible GPUs;
- `generate_inner_vllm` explicitly uses `SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, stop_token_ids=None)`;
- default `max_new_tokens` is 2048;
- normal VLMEvalKit returns only `o.outputs[0].text` and discards `o.outputs[0].token_ids`;
- `VStarBench` is an `ImageMCQDataset` and follows the standard dataset MCQ prompt path;
- pinned `Qwen2.5-VL-7B-Instruct` registration sets `use_custom_prompt=False`;
- Monet's system prompt is inserted independently by `Qwen2VLChat` before the user content.

## VStarBench Dataset / Prompt Plumbing — VERIFIED
The first dataset-only build completed successfully after an initial failed network route automatically fell back to a working download route.

Observed result:
```text
VStarBench.tsv: 162MB [00:16, 9.97MB/s]
dataset class: ImageMCQDataset
dataset name: VStarBench
num samples: 191
columns: ['index', 'question', 'A', 'B', 'C', 'D', 'answer', 'category', 'image']
VSTAR_DATASET_PROMPT_PASS=True
```

First sample:
```text
index: 0
question: What is the material of the glove?
A: rubber
B: cotton
C: kevlar
D: leather
answer: A
image: /home/user6/LMUData/images/VStarBench/0.png
```

First standard prompt:
```text
Question: What is the material of the glove?
Options:
A. rubber
B. cotton
C. kevlar
D. leather
Please select the correct answer from the options above.
```

This closes the dataset/network/prompt plumbing check.

## Raw-Token Probe — IMPLEMENTED, NOT YET RUN
Two scripts were added to V_COT:
- `scripts/08_vstar_single_raw_token_probe.py`
- `scripts/08_vstar_single_raw_token_probe.sh`

The launcher:
- verifies the pinned Monet commit and local checkpoint;
- creates a temporary `Monet_models` package containing the exact pinned Monet runner;
- applies the official Monet startup patch body via correctly named `sitecustomize.py`;
- sets `LATENT_SIZE=10` and latent IDs `151666/151667`;
- defaults to four low-memory GPUs, with `VCOT_GPUS` available as an explicit override;
- runs from the pinned VLMEvalKit directory and logs to `logs/08_vstar_single_raw_token_probe.log`.

The Python probe:
- builds VStarBench sample 0 through VLMEvalKit;
- instantiates `Qwen2VLChat` with the local Monet-7B checkpoint, official Monet system prompt, VStar-compatible image preprocessing, `use_custom_prompt=False`, and `use_vllm=True`;
- preserves VLMEvalKit generation semantics;
- wraps only `self.llm.generate` to capture the returned vLLM object observationally;
- records the final chat-template prompt, actual sampling parameters, raw generated text, `o.outputs[0].token_ids`, token pieces, latent start/end positions, and latent segment count;
- explicitly rejects any non-empty `allowed_token_ids`, so this probe cannot silently become another forced-token diagnostic.

The raw-token probe has not yet been executed on the GPU server, so no natural latent-trigger claim should be made yet.

## Natural Latent-Trigger Plan
First scientific characterization after the single-sample probe passes:
- dataset: `VStarBench`;
- initial subset: 20–50 examples;
- `LATENT_SIZE=10`;
- no forced tokens and no `allowed_token_ids`;
- generation semantics identical to the pinned VLMEvalKit vLLM path;
- record raw token IDs, raw text, start-token presence (`151666`), end-token presence (`151667`), and latent-segment count;
- compute `r_trigger = #samples emitting 151666 / total` only after the capture path is verified.

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
- [x] Verify VStarBench dataset build and first prompt without model loading.
- [x] Implement observational single-sample raw-token capture.
- [ ] Run and verify the single-sample raw-token probe.
- [ ] Measure natural latent-trigger frequency on a 20–50 sample VStarBench subset.
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
- `decord==0.6.0` triggers a platform-support warning in `pip check` even though Python import succeeds.
- vLLM shutdown may emit NCCL/resource-tracker cleanup warnings after successful inference.

## Next Action
Transfer/create `scripts/08_vstar_single_raw_token_probe.py` and `scripts/08_vstar_single_raw_token_probe.sh` on the GPU server, then run the launcher once. Do not modify model sampling, do not add forced token constraints, and do not start the 20–50 sample scan until the single-sample probe prints `VSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True`.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
