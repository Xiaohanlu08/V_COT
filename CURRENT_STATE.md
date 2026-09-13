# Current State

## Project Stage
V0 preparation: evidence-discriminative latent supervision.

## Current Objective
Implement the smallest architecture-preserving Stage-3 SFT experiment that tests whether task-relevant visual evidence makes Monet recurrent latent states more useful.

## Pinned Baseline
- Monet source: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Current reproduced inference checkpoint: local `models/Monet-7B`
- VLMEvalKit: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Runtime remains protected; do not modify package versions without a demonstrated blocker.

## Evidence Gate — PASSED
The frozen 12-sample VStarBench direct-attribute target-vs-sham confirmatory experiment passed:
```text
mechanically valid: 12/12
positive specificity: 10/12
specificity median: +0.0004564524
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p: 0.0009765625
```
This supports target-specific latent sensitivity under the tested full-image occlusion protocol. It does not establish answer-level utility; latent-start suppression remained null (`54/72` vs `54/72`, McNemar `p=1.0`).

## Training-Data Direction — VISUAL_COT ONLY FOR FIRST V0
`Monet-SFT-125K` has 125072 examples; `Visual_CoT` contributes 118561 (`94.7942%`). Actual-image audit showed the six helper families are heterogeneous, so the first V0 pilot must not use one universal evidence operator across all subsets.

## EXP-0015 — Visual_CoT Crop Recoverability — PASSED
Frozen 64-row cohort SHA:
```text
b375ac07747c17525187cafc5c842fed1a20a60d1174cc927d5c3f9bc9f89063
```
Verified robust Step-22b result:
```text
valid pairs: 64/64
strong recoverable: 61/64 = 0.953125
median same-source NCC: 0.9923841
median wrong-source NCC: 0.4728785
median same-minus-wrong margin: 0.5103442
crop_family_gate_passed: true
```
The official helper crop is therefore licensed as a localized task-relevant region for the first Visual_CoT V0 operator.

## EXP-0016 — Same-Source Disjoint Sham Crop — FAILED
Step 23 frozen gate failed:
```text
constructible: 44/61 = 0.7213114754
nonconstructible: 17/61
geometry violations: 0
pixel-identical negatives: 0
frozen_gate_passed: false
```
Do not rescue this operator by post-hoc filtering or threshold relaxation.

## EXP-0017 — Full-Image Recovered-Evidence Neutral Occlusion — PASSED
Step 24 used exactly the 61 Step-22b strong samples and did not use Step-23 constructibility filtering.

Operator:
1. keep the full original source image geometry;
2. recover the Step-22b evidence box;
3. expand by 0.25 box width/height to define a surrounding ring;
4. compute rounded mean RGB over the ring excluding the evidence box;
5. replace only the evidence-box pixels with that RGB.

Verified result:
```text
valid_count: 61/61
invalid_count: 0
dimension_failure_count: 0
outside_change_failure_count: 0
ring_failure_count: 0
changed_fraction_failure_count: 0
mean_abs_diff_failure_count: 0
frozen_gate_passed: true
records_sha256: 90d181db15f303dcbd38e139dd15989e8cfa3c37122d6e3cd2b6d322abb7e5bd
```
Mask-area distribution:
```text
min:    0.00212048
median: 0.097888
p90:    0.38896
max:    0.97265625
```
The high maximum area is retained as a recorded property; no post-hoc maximum-area filter is introduced.

## First V0 Data Operator — FROZEN
For eligible Visual_CoT examples:
```text
positive/student view: original full source image
negative/student view: same full image with recovered evidence region neutral-occluded
teacher target: official helper-derived cached Stage-3 teacher representation
```
The negative-image operator is now frozen. Do not return to the failed disjoint-sham operator unless new pre-specified evidence justifies it.

## Official Monet Stage-3 Facts
Official Stage 3:
- generates student recurrent latents from the student image branch;
- exposes `student_outputs_latent.ce_patch_pos` and `ce_patch_vec`;
- injects them into the second CE/alignment forward;
- loads cached teacher representations through `teacher_latent_dir`;
- optimizes `student_ce_loss + alignment_weight * alignment_loss`.

The official Stage-3 recipe precomputes teacher targets from the Stage-2 model with `--output_hidden_states` and then trains Stage 3 with `--alignment_layer all_layers`. Therefore the cached teacher target is not required to be `ce_patch_vec`; it is saved hidden-state output under the file key `latent`.

## Step 25 — STATIC AUDIT SCRIPT INVALID, NOT A MONET FAILURE
The first Step-25 audit aborted because it incorrectly required the string `ce_patch_vec` to appear inside `src/precompute_teacher_latents.py`.

That requirement was wrong. In the pinned official implementation:
```text
teacher precompute: outputs.hidden_states  (official recipe uses --output_hidden_states)
student Stage-3 latent forward: ce_patch_pos / ce_patch_vec
cached teacher file key: latent
Stage-3 alignment_layer: all_layers
```
So Step 25 produced no scientific result and does not indicate a source-code or training-path failure. Step 25b corrects only the audit specification.

## Candidate V0 Loss — NOT YET FROZEN
Conceptually, the smallest extension remains:
```text
L_V0 = L_CE + lambda_align * L_align + lambda_evidence * L_evidence
```
where the evidence term compares the original-image student latent against the evidence-occluded student latent relative to the official teacher target.

Do not yet freeze direct cosine/triplet/ranking details. The official model contains `affine_subspace_alignment_loss`, and runtime shape/nesting/reduction must be inspected before defining the evidence loss.

## Active Constraints
- No architecture change.
- V0 is SFT-only; no VLPO changes yet.
- Visual_CoT only for the first V0 operator.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Do not change the frozen Step-24 negative operator after outcome inspection.
- Do not download/rebuild Stage1/2 assets until the minimum runtime probe requirements are clear.
- Any continued-SFT V0 experiment must have a matched continued-SFT control with identical data/steps except the new evidence loss.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Audit Stage-3 hooks/environment.
- [x] Audit Monet-SFT-125K helper structure.
- [x] Validate Visual_CoT crop recoverability.
- [x] Reject universal disjoint sham crops.
- [x] Validate and freeze full-image recovered-evidence neutral occlusion.
- [ ] Complete corrected Step-25b static Stage-3 loss-contract audit.
- [ ] Determine the minimum published assets needed for a runtime tensor-shape/memory probe.
- [ ] Freeze the exact V0 evidence loss only after the runtime probe.
- [ ] Run a tiny deterministic smoke test.
- [ ] Run matched continued-SFT baseline vs V0 pilot.

## Next Action
Run Step 25b. Treat `precompute_mentions_ce_patch_vec=false` as expected: official teacher precompute uses hidden states, while `ce_patch_vec` belongs to the student Stage-3 latent forward.