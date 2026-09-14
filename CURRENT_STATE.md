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
Verified on Visual_CoT row 0 using local `models/Monet-7B`, `latent_size=8`, and the frozen Step-24 occluded negative:
```text
positive/negative input_ids: [1,334], identical
alignment positions: [308,309,310,311,312,313,314,315]
positive ce_patch_vec: list[Tensor(8,3584)] bf16
negative ce_patch_vec: list[Tensor(8,3584)] bf16
ce_patch_pos == alignment positions: true
positive/negative aligned hidden tensor: [29,8,3584]
```
Sequential inference-only memory remained below about `15.69 GiB` peak allocated on the tested GPU. This does not establish train-time feasibility because no backward/optimizer graph was present.

Synthetic self-target mechanics only:
```text
official/default-dim gap (negative-positive): ~0.07763
explicit dim=-1 hidden-vector gap: ~0.01979
```
These synthetic values are not official Stage-2 teacher measurements and must not be used to choose V0 hyperparameters.

## EXP-0020 — Public Stage-2 Teacher Asset Acquisition — VERIFIED
The only additional model asset justified after EXP-0019 was the public Stage-2 teacher used by official Stage-3 precompute. Acquired only:
```text
NOVAglow646/Monet-SFT-7B/stage2
local: models/Monet-SFT-7B-stage2
```
Stage-1 and Stage-3 were not downloaded.

The first Step-27 final equality check was invalid because it compared total safetensors container-file bytes to `model.safetensors.index.json:metadata.total_size`. Step 27b corrected verification by parsing each safetensors header and summing tensor `data_offsets` payload bytes.

Verified Stage-2 structure:
```text
index tensor payload bytes: 16578684928
parsed tensor payload bytes: 16578684928
local container file bytes: 16578766160
container/header overhead: 81232 bytes
all shard container sizes exact: true
all tensor byte ranges non-overlapping: true
per-shard index names match headers: true
global index names match: true
```
Shard SHA-256:
```text
model-00001-of-00004.safetensors  daa156afaf34fed7be870dbccdd12db0e3e92187c9d755f31653c4ccb6ce2954
model-00002-of-00004.safetensors  5140dfeab39fe95c784bc8bfd4e3279b1ff2059e376ea58aebedd3bb290e5799
model-00003-of-00004.safetensors  bd55afc3a00da7cd44099e9e0fc21a1d535a62ee039e79dacc050afc64117aab
model-00004-of-00004.safetensors  debf05227df9795774a51a1fc49e1b980e731999861565f1cb986408c4514d73
```
The Stage-2 asset is therefore structurally verified and should not be re-downloaded.

## Candidate V0 Loss — TENSOR SPACE RESOLVED, REAL TEACHER SCALE PENDING
Preserve the matched baseline exactly:
```text
L_base = L_CE + lambda_align * L_align_official
```
The evidence term should compare original and evidence-occluded student branches against the same official Stage-2 teacher target in the verified all-layer tensor space `[29, latent_count, 3584]`.

Runtime inspection supports explicit `dim=-1` hidden-vector cosine distance as a semantically clean candidate, averaged over layers and latent positions, while the official baseline `dim=1` alignment remains untouched. The exact ranking form, stop-gradient choice, margin/temperature, reduction, and `lambda_evidence` remain unfrozen until real Stage-2 teacher distances are measured.

## Active Constraints
- No architecture change.
- V0 is SFT-only; no VLPO changes yet.
- Visual_CoT only for the first V0 operator.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 negative operator unchanged.
- Preserve official Stage-3 baseline alignment exactly in matched controls.
- Do not use synthetic self-target scores to choose V0 loss hyperparameters.
- Do not re-download Stage-2; its local shards are structurally verified.
- Do not download Stage-3 for shape inspection.
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
- [x] Acquire and structurally verify the public Stage-2 teacher checkpoint.
- [ ] Measure real Stage-2 teacher alignment gaps on a small outcome-blind Visual_CoT cohort.
- [ ] Freeze exact V0 evidence loss.
- [ ] Run a tiny deterministic training smoke test.
- [ ] Run matched continued-SFT baseline vs V0 pilot.

## Next Action
Run a small fixed-cohort real-teacher probe: generate official Stage-2 all-layer teacher targets using the exact precompute path, then compare original and frozen evidence-occluded student tensors from local `Monet-7B` using both the official default-dim alignment and explicit `dim=-1` hidden-vector distance. Use the result to choose the evidence metric family, not to claim benchmark improvement.