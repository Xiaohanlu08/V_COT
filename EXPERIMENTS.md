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
```

**Interpretation:** the visual intervention changes the model's latent-entry policy, but the textual pre-latent trajectories and trigger timing differ, so direct latent similarity from the natural run is not interpretable.

**Conclusion:** KEEP as an alignment-failure result.

---

## EXP-0007 — Fixed-prefix / fixed-trigger crop-based evidence replay on VStarBench position 0
**Status:** COMPLETED; MECHANICALLY VALID, INITIAL EVIDENCE-SEPARABILITY HYPOTHESIS NOT SUPPORTED

**Scripts:** `scripts/14_make_fixed_trigger_runner.py`, `scripts/14_vstar_fixed_prefix_evidence_replay.py`, `scripts/14_vstar_fixed_prefix_evidence_replay.sh`

**Purpose:** Remove textual-prefix and latent-entry timing confounds from EXP-0006 and test the original single-sample hypothesis `sim(z(I),z(I+)) > sim(z(I),z(I-))`.

**Replay protocol:**
- exact 25 baseline generated token IDs before the natural latent-start token are appended directly as `prompt_token_ids`;
- the base prompt token IDs and chat-template text are identical across original, positive-crop, and negative-crop views;
- a temporary runner patch overrides only the first sampled token with `151666=<abs_vis_token>`;
- all subsequent sampling remains unchanged;
- recurrent `st["pending"]` tensors are captured exactly as in EXP-0005.

**Mechanical validation:**
```text
prefix_generated_token_count: 25
prefix_decode_encode_roundtrip_exact: true
base_prompt_ids_identical_across_views: true
chat_template_text_identical_across_views: true
original: first_token=151666, segment=(0,10), latent_shape=(10,3584)
positive: first_token=151666, segment=(0,10), latent_shape=(10,3584)
negative: first_token=151666, segment=(0,10), latent_shape=(10,3584)
mechanical_alignment_pass: true
```

**Original replay reproduction gate against EXP-0005:**
```text
mean cosine: 0.9999061822891235
min cosine: 0.9998059272766113
max cosine: 0.9999465346336365
mean relative L2: 0.013379891403019428
valid_for_counterfactual_comparison: true
```

**Evidence similarity:**
```text
S_positive_mean: 0.9516485333442688
S_negative_mean: 0.9566512107849121
Delta_evidence = S_positive - S_negative: -0.005002707242965698
relative L2 original-positive: 0.3006730079650879
relative L2 original-negative: 0.2855320870876312
mean cosine positive-negative: ~0.98892948
```

Step-wise `Delta_evidence` was positive only on step 1 and negative on 9/10 steps.

**Interpretation:** The replay protocol is valid, but the initial crop-based evidence-separability hypothesis is not supported on this sample. The negative delta must not be reframed as evidence for visual grounding. Because `I+` and `I-` share the same crop/resize geometry and are highly similar to each other, while both differ much more from the full original image, the dominant perturbation appears to be crop/resize geometry rather than target evidence removal.

**Conclusion:** KEEP as a negative single-sample result and as validation of the fixed-prefix/fixed-trigger instrumentation. Do not start V0 from this evidence.

---

## EXP-0008 — Full-image target occlusion specificity on VStarBench position 0
**Status:** COMPLETED; PROMISING SINGLE-SAMPLE TARGET-SPECIFIC SIGNAL

**Scripts:** `scripts/15_vstar_target_occlusion_specificity.py`, `scripts/15_vstar_target_occlusion_specificity.sh`

**Purpose:** Remove the crop/resize confound identified in EXP-0007 and test whether occluding the benchmark-annotated target region perturbs Monet's aligned recurrent latent trajectory more than matched nuisance occlusions elsewhere in the same image.

**Design:**
- all conditions preserve the full `2000 x 1500` image geometry;
- target intervention neutral-masks the official glove bbox `[564,142,155,157]`;
- 32 deterministic same-size sham masks are placed elsewhere in the image while excluding the target and nearby target context;
- target and sham masks use the same surrounding-ring mean-RGB fill operator;
- all runs use the validated fixed 25-token prefix and one-time forced latent start from EXP-0007;
- original-image replay must pass the EXP-0005 reproduction gate before interpretation.

**Primary statistic:**
```text
cosine_distance(mask) = 1 - mean_t cosine(z_t(original), z_t(masked))
specificity = target_distance - median(sham_distances)
```

**Results:**
```text
original_replay_valid: true
target_mean_cosine: 0.9989594221115112
target_cosine_distance: 0.0010405778884887695
target_mean_relative_l2: 0.043955929577350616
control_cosine_distance_median: 0.00048357248306274414
control_cosine_distance_mean: 0.0005317628383636475
specificity_score: 0.0005570054054260254
target_distance / median-control distance: ~2.1519x
target_distance_percentile_among_controls: 96.875
num_controls_with_distance_ge_target: 1/32
empirical_one_sided_p: 0.06060606060606061
VSTAR_TARGET_OCCLUSION_SPECIFICITY_PASS=True
```

**Interpretation:** On this single annotated sample, masking the task-relevant glove region perturbs the aligned latent trajectory about `2.15x` more than the median matched sham mask. Only one of 32 sham masks has an equal-or-larger cosine-distance perturbation, placing the target at the `96.875`th percentile of the within-image control distribution. This is directionally consistent with target-specific visual sensitivity.

The empirical one-sided p-value is `(1 + 1)/(32 + 1) = 0.0606`, so the result is not conventionally significant at `0.05`. More importantly, this is only one sample. Increasing the sham count on position 0 alone would refine the within-image null but would not establish generalization across questions/images.

**Conclusion:** KEEP as a promising single-sample result. It justifies multi-sample replication, but it does not yet justify claiming visual grounding or starting V0.

---

## EXP-0009 — Triggered-72 official V*Bench annotation mapping audit
**Status:** COMPLETED

**Scripts:** `scripts/16_vstar_annotation_mapping_audit.py`, `scripts/16_vstar_annotation_mapping_audit.sh`

**Purpose:** Establish a reproducible mapping from the 72 naturally-triggered VLMEvalKit VStarBench positions to official `craigwu/vstar_bench` annotations before selecting any multi-sample target-occlusion replication cohort.

**Mapping rule:** category + normalized exact question; repeated question/category matches are disambiguated by option-string compatibility. No target-occlusion outcomes are used.

**Results:**
```text
triggered_total: 72
official_annotation_file_count: 191
mapped_count: 69
unresolved_count: 3
unique_question_category: 62
question_category_plus_options: 7
mapped direct_attributes: 41
mapped relative_position: 28
single_target_single_bbox_count: 49
single_target_single_bbox direct_attributes: 41
single_target_single_bbox relative_position: 8
unresolved_positions: [29, 49, 61]
VSTAR_TRIGGERED72_ANNOTATION_AUDIT_PASS=True
```

The unresolved cases are repeated direct-attribute questions:
```text
29: What is the color of the man's cap?       candidates=4
49: What is the color of the plastic stool?   candidates=2
61: What is the color of the dog?             candidates=3
```

**Interpretation:** There is a sufficiently large clean pool for outcome-blind replication. All 41 mapped direct-attribute samples have exactly one target object and one bbox, while only 8 relative-position samples satisfy that criterion. Because relative-position questions require relational evidence and often multiple objects/context, mixing them into the same target-only occlusion cohort would blur the intervention semantics.

**Conclusion:** KEEP. The confirmatory replication cohort will therefore be selected only from mapped, naturally-triggered, single-target/single-bbox `direct_attributes` samples; development position 0 will be excluded. The unresolved positions are excluded rather than manually resolved after observing specificity outcomes.

**Next action:** freeze a 12-sample direct-attribute replication cohort with a fixed seed and persist its SHA-256 before running any additional GPU target-occlusion inference.

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
