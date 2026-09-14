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
Frozen 12-sample VStarBench direct-attribute target-vs-sham confirmatory result:
```text
mechanically valid: 12/12
positive specificity: 10/12
specificity median: +0.0004564524
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p: 0.0009765625
```
This supports target-specific latent sensitivity under the tested full-image occlusion protocol. It does not establish answer-level utility; latent-start suppression remained null (`54/72` vs `54/72`, McNemar `p=1.0`).

## First V0 Training Family — Visual_CoT
`Monet-SFT-125K` has 125072 examples; `Visual_CoT` contributes 118561 (`94.7942%`). The six helper-image families are heterogeneous, so the first V0 pilot is restricted to Visual_CoT.

## EXP-0015 — Visual_CoT Crop Recoverability — PASSED
Frozen 64-row cohort SHA:
```text
b375ac07747c17525187cafc5c842fed1a20a60d1174cc927d5c3f9bc9f89063
```
Verified robust result:
```text
valid pairs: 64/64
strong recoverable: 61/64 = 0.953125
median same-source NCC: 0.9923841
median wrong-source NCC: 0.4728785
median same-minus-wrong margin: 0.5103442
crop_family_gate_passed: true
```

## EXP-0016 — Same-Source Disjoint Sham Crop — FAILED
Frozen Step-23 gate:
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
5. replace only evidence-box pixels with that RGB.

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
min: 0.00212048
median: 0.097888
p90: 0.38896
max: 0.97265625
```
No post-hoc maximum-area filter is introduced.

## First V0 Data Operator — FROZEN
For eligible Visual_CoT examples:
```text
positive/student view: original full source image
negative/student view: same full image with recovered evidence region neutral-occluded
teacher target: official helper-derived cached Stage-3 teacher representation
```

## EXP-0018 — Corrected Static Stage-3 Loss Contract Audit — PASSED
Verified on the pinned Monet source:
- teacher precompute uses Stage-2 `outputs.hidden_states` with `--output_hidden_states` and saves them under dictionary key `latent`;
- official Stage 3 uses `--alignment_layer all_layers`;
- student latent forward generates `ce_patch_pos` and `ce_patch_vec`;
- `ce_patch_vec[b]` is `Tensor(num_latents_b, H)` and is injected into latent-token positions for the second forward;
- `student_alignment_poss` are latent-pad positions;
- official second-forward alignment uses all-layer hidden states at those positions;
- official objective is `student_ce_loss + alignment_weight * alignment_loss`;
- no negative-image branch exists;
- `affine_subspace_alignment_loss` is defined but is not the active Stage-3 objective;
- pinned all-layer `alignment_loss` calls `cosine_similarity(...)` without explicit `dim`, so the active reduction uses PyTorch default `dim=1` on `[layers, align_positions, hidden_dim]`.

## EXP-0019 — Stage-3 Runtime Contract Probe — PASSED
Step 26 first established the latent-forward contract but aborted only in its synthetic hidden-state diagnostic because the probe omitted `alignment_poss`. Step 26b corrected only that probe plumbing and completed successfully.

Verified runtime contract on Visual_CoT row 0 using existing local `models/Monet-7B`, `latent_size=8`, and the frozen Step-24 occluded negative:
```text
positive input_ids: [1,334]
negative input_ids: [1,334]
token IDs identical: true
alignment positions: [308,309,310,311,312,313,314,315]
positive ce_patch_vec: list[Tensor(8,3584)] bf16
negative ce_patch_vec: list[Tensor(8,3584)] bf16
positive ce_patch_pos == negative ce_patch_pos == alignment positions: true
aligned hidden tensor: [29,8,3584]
positive/negative aligned hidden shapes identical: true
```
Sequential inference-only memory/runtime on the tested GPU:
```text
model baseline allocated: 15.4875 GiB
positive latent peak allocated: 15.5637 GiB
negative latent peak allocated: 15.5807 GiB
positive hidden-forward peak allocated: 15.6453 GiB
alignment probe peak allocated: 15.6853 GiB
```
These values do not establish train-time feasibility because no backward/optimizer graph was present.

Synthetic self-target mechanics only:
```text
official/default-dim alignment gap (negative-positive): ~0.07763
explicit dim=-1 hidden-vector gap: ~0.01979
```
The synthetic target is not the official Stage-2 teacher and these values must not be used to select a margin or make a scientific claim.

## Candidate V0 Loss — TENSOR SPACE RESOLVED, SCALE NOT YET FROZEN
Preserve the matched baseline exactly:
```text
L_base = L_CE + lambda_align * L_align_official
```
The additional evidence term should compare the original and evidence-occluded branches against the same official Stage-2 teacher target in the verified all-layer tensor space `[29, latent_count, 3584]`.

Runtime inspection now supports an explicit per-hidden-vector cosine along `dim=-1` as a semantically clean candidate evidence distance, averaged over layer and latent position, while the official baseline alignment remains unchanged. However, the exact ranking form, stop-gradient choice, margin/temperature, reduction, and `lambda_evidence` are not frozen because real Stage-2 teacher distances/gradients have not yet been measured.

## Asset Decision After Step 26b
- The public Stage-3 checkpoint is **not needed** for shape/mechanics inspection.
- Existing local `Monet-7B` was sufficient to establish the runtime contract.
- To measure real official teacher distances and calibrate the evidence-loss scale, the next justified model asset is the public `Monet-SFT-7B/stage2` checkpoint (teacher used by official Stage-3 precompute), not Stage-3.
- Do not rebuild Stage1 or full teacher caches; a single-sample direct Stage-2 teacher probe is sufficient first.

## Active Constraints
- No architecture change.
- V0 is SFT-only; no VLPO changes yet.
- Visual_CoT only for the first V0 operator.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 negative operator unchanged.
- Preserve official Stage-3 baseline alignment exactly in matched controls.
- Do not use synthetic self-target scores to choose V0 loss hyperparameters.
- Any continued-SFT V0 experiment must have a matched continued-SFT control with identical data/steps except the new evidence loss.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Audit Stage-3 hooks/environment.
- [x] Audit Monet-SFT-125K helper structure.
- [x] Validate Visual_CoT crop recoverability.
- [x] Reject universal disjoint sham crops.
- [x] Validate and freeze full-image recovered-evidence neutral occlusion.
- [x] Complete corrected static Stage-3 loss-contract audit.
- [x] Complete local runtime shape/mechanics probe.
- [ ] Acquire only the public Stage-2 SFT checkpoint needed for a real teacher-target probe.
- [ ] Measure real positive/negative teacher-alignment distances and gradient/memory behavior on a tiny fixed sample.
- [ ] Freeze exact V0 evidence loss.
- [ ] Run a tiny deterministic smoke test.
- [ ] Run matched continued-SFT baseline vs V0 pilot.

## Next Action
Download only `NOVAglow646/Monet-SFT-7B/stage2` with a resumable, integrity-checked path, then run a single-sample real Stage-2 teacher-target probe. Do not download Stage-3.