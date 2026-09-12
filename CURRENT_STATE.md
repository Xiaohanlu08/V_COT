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
Verified on 2026-09-12/13:
- Monet source is present at `~/work/V_COT/third_party/Monet` and HEAD is exactly `08939998d3d643a73a316e349faa34f420429153`.
- `vcot` Conda environment exists with Python 3.10.21.
- Server has 10 x RTX 3090, 24 GiB each; driver 570.144.
- Protected runtime remains `torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`, `vllm==0.10.0`, `transformers==4.54.0`, `trl==0.15.2`.
- Additional protected packages: `huggingface-hub==0.36.2`, `tokenizers==0.21.4`, `accelerate==1.15.0`, `datasets==5.0.1`, `qwen-vl-utils==0.0.14`.
- Monet-7B checkpoint is fully downloaded and structurally verified.
- Pinned VLMEvalKit imports successfully as version `0.2rc1`.

## Verified Baseline Inference
- Official Monet example runs successfully and returns `\\boxed{C}`.
- `<abs_vis_token>` -> `151666`.
- `</abs_vis_token>` -> `151667`.
- `scripts/06_official_runner_check.sh` passed in parent and spawned child processes.
- `scripts/07_official_forced_latent_path.sh` passed; deterministic forced latent IDs matched `[151666, 151666, 151667]` with `LATENT_SIZE=2`.
- Forced-token checks are engineering diagnostics only, not benchmark evidence.

## VLMEvalKit Dependency Bring-Up — COMPLETE
A full unconstrained `pip install -r requirements.txt` was intentionally avoided because it would perturb the verified Monet/Hugging Face stack. Controlled bring-up used a filtered dependency set plus constraints. `rouge==1.0.1` was added as the only undeclared hard import found by the static import audit. `setuptools==81.0.0` restores the legacy `pkg_resources` API required by `openai-clip`.

Non-fatal messages may still appear:
- missing `.env` warning;
- Jieba `pkg_resources` deprecation warning;
- Transformers `TRANSFORMERS_CACHE` deprecation warning;
- `pip check` may report `decord 0.6.0 is not supported on this platform`; this is not a VStarBench blocker.

Dependency bring-up is closed. Do not install additional optional packages unless a selected evaluation path demonstrably requires them.

## Official Monet Evaluation Integration Facts
Pinned Monet README specifies:
- `vllm==0.10.0`;
- the Monet inference runner must replace `vllm.v1.worker.gpu_model_runner`;
- latent IDs are `151666/151667`;
- official evaluation system prompt: `You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}.`;
- an API model is used as supplementary judge for exact reproduction of reported benchmark scores.

V_COT uses the official startup-patch body with the filename corrected from the README typo `sitecustomized.py` to Python's real autoload hook `sitecustomize.py`.

## Pinned VLMEvalKit Qwen2.5-VL / VStar Path — INSPECTED
- wrapper: `Qwen2VLChat`;
- vLLM construction: `max_num_seqs=5`, `max_model_len=32768`, image limit 24, GPU memory utilization 0.9, automatic TP from visible GPUs;
- actual vLLM sampling: `temperature=0.0`, `max_tokens=self.max_new_tokens`, `stop_token_ids=None`;
- default `max_new_tokens=2048`;
- normal wrapper returns only `o.outputs[0].text` and discards `o.outputs[0].token_ids`;
- `VStarBench` is `ImageMCQDataset` and follows the standard dataset MCQ prompt path;
- Monet system prompt is inserted independently before user content.

## VStarBench Dataset / Prompt Plumbing — VERIFIED
Dataset build succeeded:
```text
dataset class: ImageMCQDataset
dataset name: VStarBench
num samples: 191
columns: ['index', 'question', 'A', 'B', 'C', 'D', 'answer', 'category', 'image']
VSTAR_DATASET_PROMPT_PASS=True
```
The first prompt is the standard image + MCQ question/options + `Please select the correct answer from the options above.` path.

## Single-Sample Natural Latent Probe — VERIFIED
`scripts/08_vstar_single_raw_token_probe.py` and `.sh` were executed successfully on VStarBench sample/index 0 through the pinned VLMEvalKit Qwen2VL/vLLM path.

Verified generation conditions:
```text
temperature: 0.0
max_tokens: 2048
allowed_token_ids: None
LATENT_SIZE=10
```
Observed:
```text
num_generated_tokens: 99
start_positions: [25]
end_positions: [35]
segments: [(25, 35)]
NATURAL_LATENT_TRIGGER=True
VSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True
```
This is direct unforced runtime evidence that Monet can naturally enter latent mode on VStarBench. It is recorded as `EXP-0001`.

## 20-Sample Natural Trigger Pilot — VERIFIED
`scripts/09_vstar_natural_trigger_scan.py` and `.sh` were executed with:
- `VCOT_N=20`;
- `VCOT_SEED=20260913`;
- `LATENT_SIZE=10`;
- greedy decoding (`temperature=0.0`);
- no forced-token constraints;
- official Monet system prompt and pinned Monet runner;
- pinned VLMEvalKit Qwen2VL/vLLM generation path.

Selected positions:
```text
[13, 14, 24, 34, 41, 48, 53, 59, 85, 93, 95, 96, 101, 107, 126, 127, 138, 159, 176, 177]
```

Observed summary:
```text
triggered_samples: 8 / 20
trigger_rate: 0.40
balanced_marker_samples: 20 / 20
multi_segment_samples: 0 / 20
total_latent_segments: 8
mean_generated_tokens: 53.1
heuristic_option_correct_count: 14 / 20
heuristic_option_accuracy: 0.70
VSTAR_NATURAL_TRIGGER_SCAN_PASS=True
```

Interpretation:
- Natural latent activation is common enough in this pilot to justify benchmark-wide characterization.
- All samples have balanced latent start/end marker accounting.
- Every triggered sample has one latent segment; no multi-segment behavior was observed in the 20-sample subset.
- With only 20 samples, the trigger-rate estimate is still imprecise; an approximate 95% Wilson interval for 8/20 is about 0.22–0.61.
- The 0.70 option accuracy is diagnostic extraction only, not an official Monet/VLMEvalKit VStarBench score.
- The pilot alone does not establish whether latent triggering is associated with category, correctness, or response length.

This run is recorded as `EXP-0002` in `EXPERIMENTS.md`.

## Next Milestones
- [x] Select and pin Monet upstream implementation.
- [x] Reproduce runtime environment.
- [x] Download and verify Monet-7B checkpoint.
- [x] Reproduce official-example inference.
- [x] Verify tokenizer latent special-token IDs.
- [x] Verify official-style runner patch in parent and spawned child.
- [x] Exercise official latent runtime path with deterministic forced diagnostic.
- [x] Place reproducible VLMEvalKit snapshot and verify VStarBench.
- [x] Complete controlled VLMEvalKit dependency bring-up.
- [x] Inspect Monet/VLMEvalKit evaluation path and raw-token capture point.
- [x] Verify VStarBench dataset build and prompt.
- [x] Verify unforced single-sample natural latent trigger.
- [x] Run reproducible 20-sample VStarBench natural-trigger pilot.
- [ ] Inspect the 20-sample JSONL for triggered vs non-triggered correctness/category/length patterns and any instrumentation anomaly.
- [ ] Expand natural-trigger characterization to all 191 VStarBench samples if the JSONL inspection is clean.
- [ ] Reproduce selected Monet benchmark baseline under documented settings.
- [ ] Freeze reproduced baseline with a Git tag.
- [ ] Locate and instrument exact latent-state tensors needed for V0 experiments.
- [ ] Start V0 only after the above checks pass.

## Known Issues
- GitHub access from the GPU server is unavailable; source synchronization must use local staging or direct file handoff.
- The VLMEvalKit snapshot was transferred as an archive and has no `.git`; the selected snapshot commit is documented here instead.
- Windows-to-Linux transfers may convert LF to CRLF.
- vLLM shutdown may emit NCCL/resource-tracker cleanup warnings after successful inference.

## Next Action
Do not start V0 training yet. First inspect `results/natural_trigger/vstar_n20_seed20260913.jsonl` for triggered/non-triggered correctness, category distribution, output length, marker balance, and any malformed latent segment. If clean, run the identical observational protocol on all 191 VStarBench samples to obtain the benchmark-wide natural trigger rate and category-level trigger statistics.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
