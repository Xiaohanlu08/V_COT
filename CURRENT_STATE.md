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
The negative-image operator is frozen. Do not return to the failed disjoint-sham operator unless new pre-specified evidence justifies it.

## EXP-0018 — Corrected Stage-3 Loss Contract Audit — PASSED
Step 25 initially aborted because the audit incorrectly required `ce_patch_vec` to appear in `precompute_teacher_latents.py`. That requirement was invalid and produced no scientific result.

Step 25b corrected only the audit specification and passed on the pinned Monet SHA. Verified static contracts:
- teacher precompute uses the Stage-2 model with `--output_hidden_states`;
- precompute reads `outputs.hidden_states` and saves each sample under dictionary key `latent`;
- official Stage 3 trains with `--alignment_layer all_layers`;
- student latent forward generates `ce_patch_pos` and `ce_patch_vec` from the student-image branch;
- `ce_patch_vec[b]` is returned as `Tensor(num_latents_b, H)` and is injected into the corresponding latent-token positions in the second forward;
- `student_alignment_poss` are the latent-pad positions in the Stage-3 student sequence;
- the second forward computes all-layer hidden states at those positions and compares them with cached teacher hidden states using mean cosine distance;
- official Stage-3 objective remains `student_ce_loss + alignment_weight * alignment_loss`;
- no official negative-image branch exists.

Important correction: `affine_subspace_alignment_loss` exists in the model source, but source search finds no official Stage-3 call site. The active Stage-3 alignment path calls `alignment_loss(...)`, i.e. all-layer cosine alignment at the latent positions. Do not treat affine-subspace alignment as the official Stage-3 training objective.

## Candidate V0 Loss — NARROWED, NOT YET FROZEN
The evidence term should be defined in the same representation space as the official Stage-3 teacher alignment rather than directly assuming cosine similarity on raw `ce_patch_vec`.

The most faithful candidate is an alignment-ranking term:
```text
D_pos = official all-layer teacher-alignment distance for original-image branch
D_neg = corresponding teacher-alignment distance for evidence-occluded branch
L_evidence = ranking/gating function of D_pos versus D_neg
```
with the original objective preserved:
```text
L_V0 = L_CE + lambda_align * D_pos + lambda_evidence * L_evidence
```
The exact ranking form, stop-gradient choice, margin/temperature, reduction, and weight are not frozen.

## Runtime Probe Requirements
Before freezing `L_evidence`, verify at runtime on the smallest feasible local path:
- exact type/nesting/shape of `student_outputs_latent.ce_patch_vec`;
- exact type/nesting/shape of cached/teacher-style all-layer targets;
- exact relation among latent count, `student_alignment_poss`, and teacher alignment positions;
- scalar/reduction behavior of the active all-layer cosine alignment;
- peak GPU memory and runtime for adding an occluded negative latent branch;
- whether a no-grad negative branch is enough for the first V0 pilot or a fully differentiable negative branch is feasible on 24 GB GPUs.

## Active Constraints
- No architecture change.
- V0 is SFT-only; no VLPO changes yet.
- Visual_CoT only for the first V0 operator.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Do not change the frozen Step-24 negative operator after outcome inspection.
- Do not mistake the unused affine-subspace helper for the official Stage-3 alignment path.
- Do not download/rebuild Stage1/2 assets before proving they are required for the runtime probe.
- Any continued-SFT V0 experiment must have a matched continued-SFT control with identical data/steps except the new evidence loss.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Audit Stage-3 hooks/environment.
- [x] Audit Monet-SFT-125K helper structure.
- [x] Validate Visual_CoT crop recoverability.
- [x] Reject universal disjoint sham crops.
- [x] Validate and freeze full-image recovered-evidence neutral occlusion.
- [x] Complete corrected Step-25b static Stage-3 loss-contract audit.
- [ ] Run the minimum local runtime shape/memory probe using existing assets first.
- [ ] Decide whether any published SFT checkpoint/teacher asset download is actually required.
- [ ] Freeze the exact V0 evidence loss only after the runtime probe.
- [ ] Run a tiny deterministic smoke test.
- [ ] Run matched continued-SFT baseline vs V0 pilot.

## Next Action
Use the existing local `models/Monet-7B` and already extracted Visual_CoT sample assets for the smallest runtime probe possible. Do not download the 16.6 GB Stage-2/Stage-3 checkpoint merely to inspect tensor shapes; only escalate to those assets if the local architecture-identical probe cannot establish the required runtime contract.