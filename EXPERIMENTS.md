# Experiment Log

This file is the permanent record of experiments that were actually run. Planned experiments stay in `CURRENT_STATE.md` until execution begins.

## Common Reproducibility Baseline
Unless an experiment explicitly states otherwise:
- Branch: `main`
- Base checkpoint: local `models/Monet-7B`
- Upstream Monet: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- VLMEvalKit snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Date: 2026-09-13
- Monet latent IDs: start `151666`, end `151667`

---

## EXP-0000 — Repository Initialization
**Status:** COMPLETED

Established `PROJECT_GOAL.md`, `CURRENT_STATE.md`, `DECISIONS.md`, and `EXPERIMENTS.md` before modifying model behavior.

---

## EXP-0001 — VStarBench single-sample natural latent-trigger probe
**Status:** COMPLETED

**Purpose:** Verify unforced natural latent activation on the real VLMEvalKit VStarBench path while retaining raw vLLM token IDs.

**Settings:** position 0, `LATENT_SIZE=10`, greedy decoding, no forced tokens.

**Result:** one natural segment from generated-token position `25` to `35`; final answer matched ground truth.

**Conclusion:** KEEP.

---

## EXP-0002 — VStarBench 20-sample natural-trigger pilot
**Status:** COMPLETED

**Data:** 20 samples without replacement, seed `20260913`.

**Result:** triggered `8/20=0.40`; all markers balanced; no multi-segment behavior; all latent segments length 10.

**Conclusion:** KEEP for trigger characterization. Correctness from the original simple parser was later superseded because semantic boxed answers were frequently misparsed.

---

## EXP-0003 — Full VStarBench natural latent-trigger characterization
**Status:** COMPLETED

**Scripts:** `scripts/09_vstar_natural_trigger_scan.py`, `scripts/09_vstar_natural_trigger_scan.sh`

**Data:** all 191 VStarBench examples.

**Settings:** `LATENT_SIZE=10`, greedy decoding, official Monet system prompt, no forced tokens.

**Results:**
```text
triggered: 72/191 = 0.3769633508
balanced markers: 191/191
multi-segment samples: 0/191
all 72 latent segments length 10
```

**Conclusion:** KEEP. Natural latent activation is substantial and reproducible.

---

## EXP-0004 — Paired latent-start suppression on naturally-triggered examples
**Status:** COMPLETED

**Scripts:** `scripts/10_vstar_latent_off_ablation.py`, `scripts/10_vstar_latent_off_ablation.sh`, `scripts/11_rescore_paired_outputs.py`, `scripts/11_rescore_paired_outputs.sh`

**Purpose:** Test the causal effect of allowing Monet to enter latent mode on the exact 72 examples that naturally triggered it.

**Intervention:** vLLM `allowed_token_ids` permits every model-vocabulary token except exactly `151666=<abs_vis_token>`. Vocabulary size `151670`; allowed IDs `151669`.

**Intervention integrity:**
```text
exact_exclusion_verified_samples: 72/72
block_verified_samples: 72/72
```

The first simple option parser produced a spurious apparent benefit for latent access. Manual auditing showed widespread semantic-answer parsing failures, including non-`None` errors. All 72 paired raw outputs were therefore rescored deterministically with actual VStarBench option strings.

**Robust local rescoring:**
```text
baseline correct:   54/72 = 0.750000
latent-off correct: 54/72 = 0.750000
delta: 0.000000
CC=50, CW=4, WC=4, WW=14
McNemar exact two-sided p = 1.0
unresolved baseline: 0
unresolved latent-off: 2 (positions 0, 34)
```

**Conclusion:** KEEP as a null answer-level causal result. No systematic correctness effect of latent entry is detected at this scale. Do not use trigger status or latent-path access alone as evidence that a latent state is useful.

---

## EXP-0005 — Exact recurrent latent tensor capture on VStarBench position 0
**Status:** COMPLETED

**Scripts:** `scripts/12_make_tensor_capture_runner.py`, `scripts/12_vstar_single_latent_tensor_probe.py`, `scripts/12_vstar_single_latent_tensor_probe.sh`

**Purpose:** Capture the exact recurrent hidden-state sequence used by Monet during a naturally triggered latent segment while proving observation-only instrumentation.

**Instrumentation:** temporary official Monet runner patched immediately after:
```python
st["pending"] = last_token_h[i].detach()
```
These are the vectors later written into `self.inputs_embeds` for the next latent decode step. Only rank-0 float32 CPU copies are saved; the live tensor is untouched.

**Validation:** complete generated token sequence matched the saved natural baseline exactly.

**Results:**
```text
generation_exact_match_baseline: true
num_latent_tensors: 10
hidden_size: 3584
dtype_saved: torch.float32
all_finite: true
mean_l2_norm: 288.1265869140625
min_l2_norm: 275.2210998535156
max_l2_norm: 293.09039306640625
adjacent_cosine_mean: 0.9219153655899895
adjacent_cosine_min: 0.6129838228225708
adjacent_cosine_max: 0.9985483288764954
```

**Saved reference:** `results/latent_capture/vstar_pos0_latents.pt`

**Conclusion:** KEEP. Exact recurrent latent-state access is mechanically verified.

---

## EXP-0006 — Natural three-view visual-evidence pilot on VStarBench position 0
**Status:** COMPLETED; NATURAL ALIGNMENT FAILED

**Scripts:** `scripts/13_vstar_evidence_pilot.py`, `scripts/13_vstar_evidence_pilot.sh`

**Purpose:** Test whether original, evidence-preserving, and evidence-destroying views naturally enter an already-aligned latent trajectory suitable for direct comparison.

**Official annotation:** `craigwu/vstar_bench/direct_attributes/sa_4690.json`.
```text
target_object: glove
bbox <x,y,w,h>: [564, 142, 155, 157]
question: What is the material of the glove?
local image size: 2000 x 1500
```

**Views:**
- `I`: original image.
- `I+`: target-centered crop using the V* reference target-patch rule with `patch_scale=1.2`, then resized to `2000 x 1500`.
- `I-`: identical crop/resize to `I+`, but the target bbox is replaced by the surrounding-ring mean RGB before resize.

**Verified view geometry:**
```text
crop_box_xyxy: [548, 126, 734, 314]
crop_size_before_resize: [186, 188]
local_target_bbox_xyxy: [16, 16, 171, 173]
neutral_fill_rgb: [137, 128, 130]
```

**Natural-run results:**
```text
original: segment [25,35], 10 latent tensors, 99 generated tokens
positive: no latent segment, 0 latent tensors, 89 generated tokens
negative: segment [16,26], 10 latent tensors, 100 generated tokens
original_generation_exact_match_baseline: true
all_three_naturally_triggered: false
all_three_have_10_latents: false
same_pre_latent_prefix_original_vs_positive: false
same_pre_latent_prefix_original_vs_negative: false
same_latent_start_position: false
aligned_natural_comparison_ready: false
VSTAR_EVIDENCE_PILOT_CAPTURE_PASS=True
```

**Observed answers:** original `A. rubber`; positive `A`; negative `D` (leather under the VLMEvalKit option ordering used for the run). These answer changes are descriptive only and are not the target metric of this experiment.

**Interpretation:** the visual intervention changes the model's latent-entry policy: the positive zoom suppresses natural latent entry, while the negative masked view triggers earlier. Because the textual pre-latent trajectories and trigger timing differ, no direct latent similarity from this natural run is scientifically interpretable.

**Conclusion:** KEEP as an alignment-failure result. Proceed to fixed-prefix / fixed-trigger counterfactual replay.

**Next action:** replay the exact 25 baseline generated tokens preceding latent start as explicit prompt token IDs, force only the first subsequent token to `151666`, capture `10 x 3584` recurrent states for `I`, `I+`, and `I-`, and first verify the original replay against EXP-0005 before interpreting evidence similarities.

---

## Experiment Template
```markdown
## EXP-XXXX — Short title
**Status:** RUNNING / COMPLETED / FAILED / ABORTED
**Date:** YYYY-MM-DD
**Branch:** `...`
**Commit:** `...`
**Base checkpoint:** `...`
**Config:** `...`
**Purpose:** ...
**Change from baseline:** ...
**Data / benchmark:** ...
**Settings:** ...
**Hardware:** ...
**Results:** ...
**Conclusion:** KEEP / REJECT / INCONCLUSIVE
**Reason:** ...
**Next action:** ...
```

## Reproducibility Rule
A result is verified only when the exact implementation, checkpoint, benchmark protocol, and numerical result are recorded here. Verified milestones should also receive a Git tag when appropriate.
