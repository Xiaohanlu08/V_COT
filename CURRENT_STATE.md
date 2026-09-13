# Current State

## Project Stage
V0 preparation: evidence-discriminative latent supervision.

## Current Objective
Turn the confirmed target-specific latent-sensitivity signal into the smallest architecture-preserving SFT experiment that tests whether task-relevant visual evidence makes Monet latent states more useful.

## Pinned Baseline
- Monet source: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Current reproduced inference checkpoint: local `models/Monet-7B`
- VLMEvalKit snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Protected runtime unchanged: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, transformers 4.54.0, trl 0.15.2, vllm 0.10.0.

## Evidence Gate — PASSED
The outcome-blind 12-sample VStarBench direct-attribute confirmatory cohort passed the frozen target-vs-sham latent-sensitivity rule:
```text
mechanically valid: 12/12
positive specificity: 10/12
specificity median: +0.0004564524
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p: 0.0009765625
```
Supported claim: under the tested full-image neutral-occlusion protocol, masking task-relevant evidence perturbs aligned recurrent Monet latent states more than matched nuisance occlusions. This is not answer-level utility; latent-start suppression remained null (`54/72` vs `54/72`, McNemar `p=1.0`).

## Monet Stage-3 Integration Facts — VERIFIED
Official Stage 3 removes assistant/helper images from student inputs, exposes `student_outputs_latent.ce_patch_vec`, uses helper-image-derived teacher latent supervision, and optimizes `student_ce_loss + alignment_weight * alignment_loss`. Therefore simply reusing helper images as a positive view is not the V0 novelty. V0 must add evidence discrimination while preserving architecture and VLPO.

## Training-Data Audits — VERIFIED
`Monet-SFT-125K` contains 125072 examples. All examples contain a user image and at least one assistant/helper image; `Visual_CoT` contains 118561 examples (`94.7942%` of the corpus).

Actual-image inspection confirmed heterogeneous helper families. `Visual_CoT` is the only family currently licensed for the first V0 pilot; the other five subsets contain annotation/highlight/state-transition/geometry transforms that require separate operators.

## EXP-0015 — Robust Visual_CoT Crop-Recoverability Audit — PASSED
The frozen 64-row cohort has SHA:
```text
b375ac07747c17525187cafc5c842fed1a20a60d1174cc927d5c3f9bc9f89063
```
The original float32 NCC implementation was invalidated and discarded. Step 22b reused the exact same rows and unchanged thresholds with float64 NCC.

Verified Step-22b result:
```text
valid pairs: 64/64
strong recoverable: 61/64 = 0.953125
strong-fraction Wilson 95% CI: [0.8710035, 0.9839310]
median same-source NCC: 0.9923841
median wrong-source NCC: 0.4728785
median same-minus-wrong margin: 0.5103442
crop_family_gate_passed: true
```
Non-strong rows: `63121`, `76645`, `107888`.

Interpretation: the dominant Visual_CoT helper family is quantitatively validated as localized source evidence. The recovered helper location can therefore be used as a task-relevant source region for V0 data construction.

## EXP-0016 — Same-Source Disjoint Sham-Crop Validation — FAILED FROZEN GATE
Step 23 evaluated the pre-specified sham rule on exactly the 61 Step-22b strong samples.

Frozen construction:
- negative crop from the same source image;
- exact recovered positive crop width/height;
- positions from a `16 x 16` lattice;
- zero overlap with the positive box expanded by `0.25 x crop width/height`;
- at least 8 valid sham positions;
- deterministic content-blind selection.

Frozen pass rule:
```text
all 61 strong samples have >= 8 valid sham positions
AND geometry violations = 0
AND pixel-identical selected negatives = 0
```

Verified result:
```text
constructible: 44/61 = 0.7213114754
nonconstructible: 17/61
valid sham count min/median/max: 0 / 128 / 254
geometry violations: 0
pixel-identical negatives: 0
frozen_gate_passed: false
```

Nonconstructible rows:
```text
7446, 10817, 21476, 33779, 43551, 64340, 67320, 68981,
87598, 90224, 94458, 103479, 105445, 106665, 109341, 113379, 114655
```
Most failures have evidence crops so large that no same-size disjoint sham can exist in the same source image. This is a structural limitation of the operator, not an implementation or transport failure.

Important logging note: `V0_SAME_SOURCE_SHAM_VALIDATION_PASS=True` means the validation script completed successfully. It does **not** override `frozen_gate_passed: false`.

## V0 Data Decision After Step 23
Do not salvage the failed sham-crop rule by post-hoc filtering to the 44 constructible examples, shrinking the exclusion region, shrinking sham size, or lowering the minimum-sham threshold. D011's exact same-source disjoint-sham operator is rejected as the universal first Visual_CoT negative.

The next candidate negative should reuse the already successful intervention family from the VStar confirmatory work:
1. use Step-22b to recover the task-relevant Visual_CoT region in the source image;
2. retain the full source image geometry;
3. neutral-fill the recovered evidence region using a surrounding-ring RGB statistic;
4. use the full-image evidence-occluded view as the negative branch.

This operator remains same-sample, preserves image geometry, is constructible even for large evidence regions, and directly avoids the crop-vs-full-image confound identified in EXP-0007. Its dataset-level mechanics still need to be validated on the frozen 61 strong samples before any training loss is frozen.

## Candidate Minimal V0 Loss — NOT YET FROZEN
If the full-image occlusion operator passes mechanical validation, the cleanest Stage-3 extension is a teacher-anchored ranking term:
```text
z_pos = student latent from original source image
z_neg = student latent from evidence-occluded source image
z_teacher = official helper-derived teacher latent

L_rank = max(0, margin - sim(z_pos, z_teacher) + sim(z_neg, z_teacher))
```
The official CE + teacher-alignment objective remains unchanged and must be present in both matched baseline and V0. This loss is only a candidate until the exact Stage-3 tensor shapes/alignment implementation and the data operator are frozen.

## V0 Design Constraints — ACTIVE
- Reuse Monet; no new architecture.
- V0 is SFT-only; VLPO remains deferred.
- Official helper-positive supervision is baseline context, not the novelty.
- V0 must add evidence discrimination.
- Do not use random cross-sample images as the primary negative.
- Do not weaken failed pre-specified gates post hoc.
- Do not mix helper transformation families under one unvalidated operator.
- Do not use the VStarBench confirmatory cohort as V0 training data.
- Do not reconstruct Stage1/2 or download all large assets before the final V0 data rule and loss are frozen.
- Do not start full V0 training yet.

## Next Milestones
- [x] Validate target-specific latent sensitivity under outcome-blind multi-sample replication.
- [x] Audit Stage-3 hooks/environment.
- [x] Audit Monet-SFT-125K helper structure.
- [x] Validate Visual_CoT as a recoverable localized-evidence family.
- [x] Test and reject the universal same-source disjoint sham-crop negative.
- [ ] Validate full-image recovered-evidence neutral occlusion on the exact 61 Step-22b strong samples.
- [ ] Freeze the first V0 data operator.
- [ ] Audit exact Stage-3 latent/alignment tensor shapes and freeze the ranking loss.
- [ ] Download only the checkpoint/data assets required by that frozen V0 pilot.
- [ ] Run a deterministic smoke test, then a matched control-vs-V0 pilot.
- [ ] Move to V1/V2 only if V0 improves the predefined matched evaluation metric.

## Next Action
Run an offline mechanical validation of full-image neutral occlusion on the exact Step-22b strong Visual_CoT cohort. Do not re-sample, change the strong criterion, or rescue the failed Step-23 operator by post-hoc filtering.