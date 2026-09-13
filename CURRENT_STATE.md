# Current State

## Project Stage
V0 preparation: evidence-discriminative latent supervision.

## Current Objective
Turn the confirmed target-specific latent-sensitivity signal into the smallest architecture-preserving SFT experiment that tests whether same-sample visual-evidence discrimination can make Monet latent states more useful.

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
negative specificity: 2/12
specificity mean:   +0.0008347084
specificity median: +0.0004564524
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p:     0.0009765625
primary_hypothesis_supported: true
```
Supported claim: under the tested full-image neutral-occlusion protocol, masking task-relevant evidence perturbs aligned recurrent Monet latent states more than matched nuisance occlusions. This is not answer-level utility or universal grounding; the earlier latent-start suppression remained null after robust rescoring (`54/72` vs `54/72`, McNemar `p=1.0`).

## Monet Stage-3 Integration Facts — VERIFIED
Official Stage 3 already removes assistant/helper images from student inputs, exposes `student_outputs_latent.ce_patch_vec`, uses helper-image-derived teacher latent supervision, and trains with `student_ce_loss + alignment_weight * alignment_loss`. Therefore simply reusing helper images as a positive view is not the V0 novelty. V0 must add evidence discrimination while keeping architecture and VLPO unchanged.

## Training-Data Audits — VERIFIED
`Monet-SFT-125K` contains 125072 examples. All examples contain a user image and at least one assistant/helper image; 124233/125072 contain `<observation>...</observation>` tags. `Visual_CoT` contains 118561 examples, or 94.7942% of the official SFT data.

Step 21 actual-image inspection confirmed heterogeneous helper families:
- `Visual_CoT`: localized crop/zoom evidence;
- `CogCoM`: same-canvas annotation/construction;
- `ReFocus`: same-canvas focus/highlight;
- `Zebra_CoT_count`: multi-step state/view transformation;
- `Zebra_CoT_visual_search`: mixed crop and rendered/boxed views;
- `Zebra_CoT_geometry`: geometry construction/decomposition.
A universal negative operator across all six subsets is rejected for V0.

## EXP-0015 — Robust Visual_CoT Crop-Recoverability Audit — PASSED
The exact frozen 64-sample cohort from Step 22 was rerun offline with numerically valid float64 NCC after the original Step-22 implementation was invalidated by impossible `|NCC| > 1` values. Selection and thresholds were not changed after that failed implementation run.

Frozen cohort:
```text
N = 64
selection_sha256 = b375ac07747c17525187cafc5c842fed1a20a60d1174cc927d5c3f9bc9f89063
```
Frozen strong rule:
```text
same-source NCC >= 0.90
AND same-minus-wrong NCC margin >= 0.05
```
Frozen family gate:
```text
all 64 pairs valid
strong fraction >= 0.90
median same-source NCC >= 0.95
median margin >= 0.10
```
Verified Step-22b result:
```text
valid pairs: 64/64
helper smaller than source: 61/64 = 0.953125
strong recoverable: 61/64 = 0.953125
strong-fraction Wilson 95% CI: [0.8710035, 0.9839310]

same-source NCC:
  mean   0.9837054
  median 0.9923841
  p10    0.9653482
  min    0.7988491

wrong-source NCC:
  mean   0.4803940
  median 0.4728785
  p90    0.6611139
  max    0.9404166

same-minus-wrong margin:
  mean   0.5033114
  median 0.5103442
  p10    0.3129825
  min   -0.0171390

crop_family_gate_passed: true
```
The three non-strong rows were `63121`, `76645`, and `107888`; all had very thin/small helper geometry. They remain in the record and are not silently removed from the audit.

## V0 Data Direction — CROP FAMILY LICENSED, NEGATIVE OPERATOR NOT YET FROZEN
Step 22b provides quantitative support that the dominant `Visual_CoT` helper family is a scalable localized crop/evidence family. This licenses validation of a same-source geometry-matched sham negative.

Candidate V0 pair construction:
1. recover the positive helper crop location in the same source image;
2. keep the official helper crop as positive evidence;
3. construct a negative crop from the same source image with the same recovered crop width/height;
4. exclude the positive region plus local context;
5. resize the negative to exactly the helper pixel dimensions;
6. choose the negative using a deterministic, content-blind rule.

This principle is preferred over cross-sample random negatives because it controls sample identity and crop geometry. It is still provisional until Step 23 passes its frozen mechanical gate.

## Step 23 — Same-Source Sham Validation — PREPARED, NOT YET RUN
Pre-outcome protocol: `protocols/V0_VISUAL_COT_SHAM_VALIDATION.md`.

Input: exactly the 61 Step-22b `strong_recoverable` samples.

Frozen negative construction:
- same source-space width/height as recovered positive box;
- candidate positions from a 16 x 16 lattice;
- positive exclusion region expanded by 0.25 crop width/height on each side;
- candidate must have zero intersection with the expanded positive region;
- at least 8 valid sham positions required;
- one negative selected by deterministic hash of `seed=20260913` and row index;
- no image content, answer text, similarity score, or model outcome is used to select the negative;
- final negative is resized to the exact helper image pixel dimensions.

Frozen Step-23 pass rule:
```text
every Step-22b strong sample has >= 8 valid sham positions
AND geometry violations = 0
AND pixel-identical selected negatives = 0
```
Passing Step 23 will freeze the first V0 `Visual_CoT` data-pair operator. It will not yet validate a contrastive loss or benchmark improvement.

## V0 Design Constraints — ACTIVE
- Reuse Monet; no new architecture.
- V0 is SFT-only; do not touch VLPO yet.
- Official helper-positive supervision is baseline context, not the novelty.
- V0 must add evidence discrimination.
- Do not use random cross-sample images as the primary negative.
- Do not mix helper transformation families under one unvalidated operator.
- Do not use the VStarBench confirmatory cohort as training data.
- Do not reconstruct Stage1/2 or download all large assets before the V0 data rule is frozen.
- Do not start full V0 training yet.

## Next Milestones
- [x] Validate target-specific latent sensitivity under outcome-blind multi-sample replication.
- [x] Audit local Stage-3 hooks/environment.
- [x] Audit complete Monet-SFT-125K metadata/helper structure.
- [x] Audit actual helper-image families across all six subsets.
- [x] Pass the robust frozen 64-sample Visual_CoT crop-recoverability gate.
- [ ] Run Step 23 and validate the same-source geometry-matched sham rule.
- [ ] If Step 23 passes, freeze the first V0 positive/negative pair operator.
- [ ] Download only the published checkpoint/data assets required by that frozen V0 pilot.
- [ ] Implement the minimal evidence-discriminative Stage-3 loss extension.
- [ ] Run a tiny deterministic smoke test, then a matched control-vs-V0 pilot.
- [ ] Move to V1/V2 only if V0 improves the predefined matched evaluation metric.

## Next Action
Run Step 23 offline on the existing Step-22b strong-recoverable samples. Do not alter the Step-22b strong labels, sham exclusion geometry, lattice size, minimum-sham count, or content-blind selection rule after seeing Step-23 outcomes.