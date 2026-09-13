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
A full unconstrained `pip install -r requirements.txt` was intentionally avoided because it would perturb the verified Monet/Hugging Face stack. Controlled bring-up used a filtered dependency set plus constraints. `rouge==1.0.1` was added as the only undeclared hard import found by the static import audit. `setuptools==81.0.0` restores legacy `pkg_resources` compatibility required by `openai-clip`.

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
Dataset build succeeded with 191 samples and the standard image + MCQ prompt path. `VSTAR_DATASET_PROMPT_PASS=True`.

## Single-Sample Natural Latent Probe — VERIFIED
`scripts/08_vstar_single_raw_token_probe.py` and `.sh` verified one unforced natural latent event on sample/index 0:
```text
LATENT_SIZE=10
temperature=0.0
allowed_token_ids=None
start_positions: [25]
end_positions: [35]
NATURAL_LATENT_TRIGGER=True
VSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True
```
Recorded as `EXP-0001`.

## 20-Sample Natural Trigger Pilot — VERIFIED
Recorded as `EXP-0002`.

Summary:
```text
triggered_samples: 8 / 20
trigger_rate: 0.40
balanced_marker_samples: 20 / 20
multi_segment_samples: 0 / 20
mean_generated_tokens: 53.1
heuristic_option_accuracy: 0.70 (diagnostic only)
```

Post-hoc subgroup inspection found triggered examples had lower diagnostic option correctness and longer outputs than non-triggered examples. These differences are exploratory and non-causal because Monet chooses when to enter latent mode; trigger status may mark harder or lower-confidence examples. Marker/segment instrumentation was clean.

## Full 191-Sample Natural Trigger Characterization — VERIFIED
`scripts/09_vstar_natural_trigger_scan.py` and `.sh` were run over all VStarBench positions `0..190` with:
- `VCOT_N=191`;
- `VCOT_SEED=20260913` (retained for protocol consistency; all samples are included);
- `LATENT_SIZE=10`;
- greedy decoding (`temperature=0.0`);
- no forced-token constraints and no `allowed_token_ids`;
- official Monet system prompt and pinned Monet runner;
- pinned VLMEvalKit Qwen2VL/vLLM evaluation path.

Observed summary:
```text
samples: 191
triggered_samples: 72
trigger_rate: 0.3769633508 (~37.7%)
approx. 95% Wilson interval: 0.311–0.447
balanced_marker_samples: 191 / 191
multi_segment_samples: 0 / 191
total_latent_segments: 72
mean_generated_tokens: 58.5916
heuristic_option_correct_count: 116 / 191
heuristic_option_accuracy: 0.60733 (~60.7%; diagnostic only)
VSTAR_NATURAL_TRIGGER_SCAN_PASS=True
```

Interpretation:
- The 20-sample pilot estimate (40%) generalized closely to the complete dataset (37.7%).
- Natural latent activation is therefore a substantial and reproducible behavior on VStarBench under the pinned evaluation path.
- Marker accounting is mechanically stable: all 191 samples have balanced start/end accounting, and all 72 triggered samples contain one latent segment; no multi-segment behavior was observed.
- The ~60.7% heuristic option accuracy is not the official Monet/VLMEvalKit VStarBench score and must not be compared directly with the paper's supplementary-API-judge result.
- No causal claim is made about trigger status versus correctness. Triggering may be an endogenous difficulty/uncertainty signal.

This run is recorded as `EXP-0003` in `EXPERIMENTS.md`.

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
- [x] Inspect the 20-sample JSONL and verify marker/segment instrumentation is clean.
- [x] Complete full 191-sample VStarBench natural-trigger characterization.
- [ ] Analyze the full JSONL by trigger status and category, including diagnostic correctness and response-length distributions.
- [ ] Reproduce the official VStarBench baseline score under Monet's documented supplementary-judge protocol.
- [ ] Freeze reproduced baseline with a Git tag.
- [ ] Locate and instrument exact latent-state tensors needed for V0 experiments.
- [ ] Start V0 only after the above checks pass.

## Known Issues
- GitHub access from the GPU server is unavailable; source synchronization must use local staging or direct file handoff.
- The VLMEvalKit snapshot was transferred as an archive and has no `.git`; the selected snapshot commit is documented here instead.
- Windows-to-Linux transfers may convert LF to CRLF.
- vLLM shutdown may emit NCCL/resource-tracker cleanup warnings after successful inference.

## Next Action
Do not start V0 training yet. First analyze `results/natural_trigger/vstar_n191_seed20260913.jsonl` over the full benchmark to quantify category-level trigger rates and triggered-vs-non-triggered differences in diagnostic correctness and response length. Then reproduce the official VStarBench score using Monet's documented supplementary API judge before freezing the baseline. After that, proceed to latent-state tensor instrumentation and the no-training positive/negative visual-evidence separability test that gates V0.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
