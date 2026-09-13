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

**Inference settings:**
- `LATENT_SIZE=10`
- `temperature=0.0`
- `max_tokens=2048`
- `allowed_token_ids=None`
- vLLM 0.10.0 with the pinned Monet runner

**Observed output:**
- final answer: `\\boxed{A. rubber}`
- ground truth: `A`
- generated tokens: 99
- latent start position: `[25]`
- latent end position: `[35]`
- one latent segment of length 10
- `NATURAL_LATENT_TRIGGER=True`
- `VSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True`

**Conclusion:** KEEP. Unforced natural latent activation is directly observable on the real VStarBench evaluation path.

**Next action:** Multi-sample characterization.

---

## EXP-0002 — VStarBench 20-sample natural latent-trigger pilot
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Script source commits:** `505f0b17590ba0105905f8dee9a671f74639733f` (Python scan) and `58dad2fc9c2d0c980b16ea7ff478e81fbde50f25` (launcher)

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Data / benchmark:** 20 VStarBench samples drawn without replacement using `VCOT_SEED=20260913`.

**Inference settings:**
- `LATENT_SIZE=10`
- greedy decoding
- no forced tokens
- official Monet evaluation system prompt

**Results:**
- triggered: `8/20 = 0.40`
- balanced markers: 20/20
- multi-segment: 0/20
- total latent segments: 8
- mean generated tokens: 53.1
- diagnostic heuristic option accuracy: 14/20 = 0.70

**Post-hoc subgroup inspection:**
- triggered (`n=8`): diagnostic correct 3/8 = 0.375; mean tokens 77.25
- non-triggered (`n=12`): diagnostic correct 11/12 = 0.917; mean tokens 37.00
- all latent segments length 10
- Fisher exact trigger vs diagnostic correctness: two-sided `p≈0.018`
- Mann–Whitney generated-token count: two-sided `p≈1.6e-5`

**Interpretation:** Trigger status is associated with longer outputs and lower diagnostic correctness in the pilot, but this is not causal evidence because trigger generation is endogenous.

**Conclusion:** KEEP. Instrumentation is clean and full-dataset characterization is justified.

**Next action:** Run all 191 VStarBench examples.

---

## EXP-0003 — Full VStarBench natural latent-trigger characterization
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Scripts:** `scripts/09_vstar_natural_trigger_scan.py`, `scripts/09_vstar_natural_trigger_scan.sh`

**Base checkpoint:** local `models/Monet-7B` from `NOVAglow646/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Characterize natural latent triggering over the complete VStarBench benchmark under the pinned, observational, unforced Monet/VLMEvalKit path.

**Change from baseline:** Raw-token observation only. No forced tokens, no `allowed_token_ids`, no change to generation semantics.

**Data / benchmark:** Complete `VStarBench`, all positions `0..190`, `n=191`.

**Inference settings:**
- `VCOT_N=191`
- `VCOT_SEED=20260913` retained for protocol consistency
- `LATENT_SIZE=10`
- latent start ID `151666`
- latent end ID `151667`
- `temperature=0.0`
- official Monet evaluation system prompt

**Primary results:**
- triggered samples: `72/191 = 0.3769633508` (~37.7%)
- balanced marker samples: 191/191
- multi-segment samples: 0/191
- total latent segments: 72
- all observed latent segments have length 10
- mean generated tokens overall: 58.5916
- diagnostic heuristic option correct: 116/191 = 0.60733
- `VSTAR_NATURAL_TRIGGER_SCAN_PASS=True`

**Full subgroup analysis:**

Triggered (`n=72`):
- diagnostic correct: `34/72 = 0.472222`
- mean generated tokens: `79.778`
- median generated tokens: `76`
- range: `50–116`
- categories: direct attributes 44, relative position 28

Non-triggered (`n=119`):
- diagnostic correct: `82/119 = 0.689076`
- mean generated tokens: `45.773`
- median generated tokens: `45`
- range: `17–94`
- categories: direct attributes 71, relative position 48

**Category-level results:**

`direct_attributes` (`n=115`):
- trigger rate: `44/115 = 0.382609`
- overall diagnostic accuracy: `81/115 = 0.704348`
- triggered diagnostic accuracy: `25/44 = 0.568182`
- non-triggered diagnostic accuracy: `56/71 = 0.788732`
- triggered mean tokens: `78.500`
- non-triggered mean tokens: `42.014`

`relative_position` (`n=76`):
- trigger rate: `28/76 = 0.368421`
- overall diagnostic accuracy: `35/76 = 0.460526`
- triggered diagnostic accuracy: `9/28 = 0.321429`
- non-triggered diagnostic accuracy: `26/48 = 0.541667`
- triggered mean tokens: `81.786`
- non-triggered mean tokens: `51.333`

**Exploratory association tests:**
- trigger status vs diagnostic correctness contingency: triggered 34 correct / 38 wrong; non-triggered 82 correct / 37 wrong
- Fisher exact odds ratio: `0.4037227214`
- Fisher two-sided p: `0.0036791367`
- Mann–Whitney U for generated-token counts: `8060.5`
- Mann–Whitney two-sided p: `1.968701622239151e-24`

**Mechanical integrity:**
- `total_segments=72`
- `segment_length_counts={10: 72}`
- `bad_segments=[]`
- `balanced_samples=191/191`
- `multi_segment_samples=0`

**Interpretation:** Natural latent triggering is a substantial and mechanically stable behavior on this benchmark. Triggered examples are much longer and have lower diagnostic option correctness than non-triggered examples, and the same directional pattern appears within both benchmark categories. The category trigger rates themselves are similar (38.3% vs 36.8%), so the overall association is not explained only by category composition. However, this is still observational: Monet chooses when to enter latent mode, so triggering may identify harder, more uncertain, or more complex trajectories rather than causing errors.

**Scientific consequence:** The result argues against using natural latent emission itself as a proxy for useful reasoning supervision. V_COT should explicitly estimate visual evidence / latent utility before supervising or distilling a latent state.

**Benchmark-score caveat:** All correctness values above come from V_COT's diagnostic boxed-option extractor. They are not Monet's official VStarBench score because the Monet README specifies a supplementary API judge.

**Conclusion:** KEEP. Full natural-trigger characterization is complete and supports moving to formal baseline-score reproduction and then causal/interventional latent-utility tests.

**Next action:** Reproduce the official VStarBench score under Monet's documented supplementary-judge protocol. Then run a paired latent-suppression ablation before V0 training. A first causal intervention should forbid only `<abs_vis_token>` while leaving all other tokens available; vLLM 0.10.0 exposes `bad_words` for this purpose, but the tokenization/blocking behavior must be verified before using it.

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
