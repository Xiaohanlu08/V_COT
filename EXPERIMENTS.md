# Experiment Log

This file is the permanent record of experiments that were actually run. Planned experiments should stay in `CURRENT_STATE.md` until execution begins.

## Recording Rules
For every experiment, record:
- experiment ID,
- date,
- branch,
- commit SHA,
- base checkpoint,
- configuration file,
- dataset / benchmark,
- training settings,
- random seed,
- hardware,
- result metrics,
- comparison baseline,
- conclusion,
- next action.

A failed experiment must be recorded as carefully as a successful one. Do not delete failed runs from project history.

---

## EXP-0000 — Repository Initialization
**Status:** COMPLETED

**Purpose:** Establish project governance and reproducibility anchors before importing or modifying model code.

**Changes:**
- Added `PROJECT_GOAL.md`.
- Added `CURRENT_STATE.md`.
- Added `DECISIONS.md`.
- Added `EXPERIMENTS.md`.

**Model run:** None.

**Metrics:** None.

**Conclusion:** Repository control structure initialized. No scientific result has been produced yet.

**Next action:** Reproduce and freeze the official Monet baseline before starting V0.

---

## EXP-0001 — VStarBench single-sample natural latent-trigger probe
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Script source commits:** `f541f02a2808b2a4230963c359276c1a01dcad39` (Python probe) and `c433e177e35f38e38bcbb22072aafe56a50483aa` (launcher)

**Base checkpoint:** local `models/Monet-7B` from `NOVAglow646/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Verify that the official Monet latent runner naturally activates on an unforced VStarBench example when driven through the pinned VLMEvalKit Qwen2VL/vLLM path, while observationally capturing raw vLLM token IDs.

**Change from baseline:** None to model generation semantics. The probe wraps `self.llm.generate` only to retain the returned vLLM object and inspect `o.outputs[0].token_ids`. No forced tokens, no `allowed_token_ids`, and no sampling modification were introduced.

**Data / benchmark:** `VStarBench`, sample position 0 / benchmark index 0.

**Prompt / target:**
- Question: `What is the material of the glove?`
- Options: A rubber / B cotton / C kevlar / D leather
- Ground truth: `A`
- Monet official system prompt was present in the final chat-template prompt.

**Inference settings:**
- `LATENT_SIZE=10`
- `temperature=0.0`
- `max_tokens=2048`
- `allowed_token_ids=None`
- vLLM 0.10.0 with the pinned Monet `monet_gpu_model_runner.py`
- tensor parallelism over four visible RTX 3090 GPUs selected by the launcher
- random seed: not applicable to greedy decoding

**Observed output:**
- final answer: `\\boxed{A. rubber}`
- ground-truth option: `A`
- generated tokens: 99
- latent start ID `151666` positions: `[25]`
- latent end ID `151667` positions: `[35]`
- detected latent segments: `[(25, 35)]`
- number of start tokens: 1
- number of end tokens: 1
- number of latent segments: 1
- `NATURAL_LATENT_TRIGGER=True`
- `VSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True`
- VLMEvalKit returned text matched raw candidate text exactly.

**Interpretation:** The first unforced VStarBench sample provides direct runtime evidence that Monet can naturally emit the latent-start token under the pinned evaluation path. The start-to-end position difference is 10, consistent with the configured `LATENT_SIZE=10` state-machine span. Text decoded from token IDs inside the latent span is not interpreted as semantic reasoning content, because Monet replaces the corresponding next-step input embeddings with cached last-layer representations during latent mode. This is a single-sample observation only and must not be reported as a benchmark-wide trigger rate.

**Conclusion:** KEEP. The observational raw-token capture path is valid enough to proceed to a multi-sample natural-trigger characterization without forced tokens.

**Next action:** Run a reproducible 20-sample VStarBench pilot, record per-sample raw token IDs/text and latent segment statistics, and compute `r_trigger = triggered_samples / total_samples`. Do not start V0 training yet.

---

## EXP-0002 — VStarBench 20-sample natural latent-trigger pilot
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Script source commits:** `505f0b17590ba0105905f8dee9a671f74639733f` (Python scan) and `58dad2fc9c2d0c980b16ea7ff478e81fbde50f25` (launcher)

**Base checkpoint:** local `models/Monet-7B` from `NOVAglow646/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Estimate the natural latent-trigger frequency on a small reproducible VStarBench subset before deciding whether to run the full 191-sample characterization.

**Change from baseline:** Observational raw-token capture only. Generation semantics remain unchanged. No forced tokens and no `allowed_token_ids` constraints.

**Data / benchmark:** `VStarBench`, 20 samples drawn without replacement from 191 total samples.

**Sampling protocol:**
- `VCOT_N=20`
- `VCOT_SEED=20260913`
- selected dataset positions: `[13, 14, 24, 34, 41, 48, 53, 59, 85, 93, 95, 96, 101, 107, 126, 127, 138, 159, 176, 177]`

**Inference settings:**
- `LATENT_SIZE=10`
- latent start ID `151666`
- latent end ID `151667`
- greedy generation through the pinned VLMEvalKit Qwen2VL/vLLM path
- no forced-token constraints
- official Monet evaluation system prompt
- hardware: V_COT GPU server with RTX 3090 GPUs; exact visible GPU IDs are recorded in the run log rather than the submitted summary

**Results:**
- samples: 20
- triggered samples: 8
- natural trigger rate: `8/20 = 0.40`
- balanced marker samples: 20/20
- multi-segment samples: 0/20
- total latent segments: 8
- mean generated tokens: 53.1
- diagnostic heuristic option correct: 14/20
- diagnostic heuristic option accuracy: 0.70
- `VSTAR_NATURAL_TRIGGER_SCAN_PASS=True`

**Post-hoc JSONL subgroup inspection:**
- Triggered (`n=8`): diagnostic correct `3/8 = 0.375`, mean generated tokens `77.25`, range `60–106`, categories `direct_attributes=5`, `relative_position=3`.
- Non-triggered (`n=12`): diagnostic correct `11/12 = 0.917`, mean generated tokens `37.00`, range `17–51`, categories `direct_attributes=9`, `relative_position=3`.
- All eight triggered samples had exactly one complete latent segment of length 10.
- All 20 samples had balanced latent start/end marker accounting; no multi-segment behavior was observed.
- Exploratory category trigger rates: direct attributes `5/14 = 0.357`; relative position `3/6 = 0.500`.
- Post-hoc Fisher exact test for trigger status vs diagnostic correctness: two-sided `p≈0.018`.
- Post-hoc Mann–Whitney test for generated-token count: two-sided `p≈1.6e-5`.

**Interpretation:** Natural latent activation is not rare in this pilot: 40% of sampled examples emitted the latent-start token under unforced greedy evaluation. The sample size is too small for a precise benchmark-wide estimate: the approximate 95% Wilson interval for 8/20 is about 0.22–0.61. The diagnostic correctness and response-length subgroup differences are strong exploratory signals, but they are not causal evidence. Monet chooses when to emit the latent-start token, so trigger status may be a marker of hard or uncertain examples rather than the cause of lower accuracy. The 0.70 heuristic option accuracy and all subgroup correctness numbers are diagnostic extraction metrics only, not official Monet/VLMEvalKit scores.

**Conclusion:** KEEP. The JSONL instrumentation is clean and the observed trigger/correctness pattern strengthens the motivation for evidence-gated or utility-gated latent supervision rather than assuming all natural latent states are useful.

**Next action:** Run the same observational protocol on all 191 VStarBench samples. Then estimate benchmark-wide trigger rate and category-level rates, and reassess trigger-vs-correctness/length associations before any causal intervention or V0 training.

---

## Experiment Template

```markdown
## EXP-XXXX — Short title
**Status:** RUNNING / COMPLETED / FAILED / ABORTED

**Date:** YYYY-MM-DD

**Branch:** `...`

**Commit:** `...`

**Tag:** `...` (if verified)

**Base checkpoint:** `...`

**Config:** `configs/...`

**Purpose:**
...

**Change from baseline:**
...

**Data / benchmark:**
...

**Training settings:**
- learning rate:
- epochs / steps:
- batch size:
- precision:
- seed:
- other:

**Hardware:**
...

**Results:**
| Metric | Baseline | This run | Delta |
|---|---:|---:|---:|
| ... | ... | ... | ... |

**Conclusion:** KEEP / REJECT / INCONCLUSIVE

**Reason:**
...

**Next action:**
...
```

## Reproducibility Rule
A result is not considered verified unless the exact commit, configuration, checkpoint, benchmark protocol, and numerical result are recorded here. Verified milestones should also receive a Git tag.
