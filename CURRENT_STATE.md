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
`Monet-SFT-125K` has 125072 examples; `Visual_CoT` contributes 118561 (`94.7942%`). The six helper-image families are heterogeneous, so the first V0 pilot remains restricted to Visual_CoT.

## Data-Operator Evidence
### EXP-0015 — Visual_CoT crop recoverability — PASSED
Frozen 64-row cohort SHA:
```text
b375ac07747c17525187cafc5c842fed1a20a60d1174cc927d5c3f9bc9f89063
```
Result:
```text
valid pairs: 64/64
strong recoverable: 61/64 = 0.953125
median same-source NCC: 0.9923841
median same-minus-wrong margin: 0.5103442
crop_family_gate_passed: true
```

### EXP-0016 — Same-source disjoint sham crop — FAILED
```text
constructible: 44/61 = 0.7213114754
nonconstructible: 17/61
frozen_gate_passed: false
```
Do not rescue this operator by post-hoc filtering or threshold relaxation.

### EXP-0017 — Full-image recovered-evidence neutral occlusion — PASSED
Step 24 used exactly the 61 Step-22b strong samples.

Operator:
1. preserve full source-image geometry;
2. recover the Step-22b evidence box;
3. expand by 0.25 box width/height for a surrounding ring;
4. compute rounded mean RGB over ring pixels outside the evidence box;
5. replace only the evidence-box pixels with that RGB.

Result:
```text
valid_count: 61/61
invalid_count: 0
all mechanical failure counts: 0
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
negative/student view: same source image with recovered evidence region neutral-occluded
teacher target: official helper-derived cached Stage-3 teacher representation
```
Do not return to the failed universal disjoint-sham operator without new pre-specified evidence.

## Stage-3 Code / Runtime Contract
### EXP-0018 — Corrected static loss-contract audit — PASSED
Verified on the pinned Monet source:
- official teacher precompute uses Stage-2 `outputs.hidden_states` with `--output_hidden_states` and saves them under key `latent`;
- official Stage 3 uses `--alignment_layer all_layers`;
- student latent forward emits `ce_patch_pos` and `ce_patch_vec`;
- `ce_patch_vec[b]` is `Tensor(num_latents_b,H)` and is injected into latent-token positions for the second forward;
- `student_alignment_poss` are latent-pad positions;
- active Stage-3 alignment calls `alignment_loss(...)`, not the unused `affine_subspace_alignment_loss` helper;
- official objective is `student_ce_loss + alignment_weight * alignment_loss`;
- pinned all-layer `alignment_loss` calls `cosine_similarity(...)` without explicit `dim`, so the active reduction uses PyTorch default `dim=1` on `[layers, align_positions, hidden_dim]`.

### EXP-0019 — Stage-3 runtime contract probe — PASSED
Verified on Visual_CoT row 0 using local `models/Monet-7B`, `latent_size=8`, and the frozen occluded negative:
```text
positive/negative input_ids: [1,334], identical
alignment positions: [308,309,310,311,312,313,314,315]
positive ce_patch_vec: list[Tensor(8,3584)] bf16
negative ce_patch_vec: list[Tensor(8,3584)] bf16
ce_patch_pos == alignment positions: true
positive/negative aligned hidden tensor: [29,8,3584]
```
Sequential inference-only peak allocation remained below about `15.69 GiB`; this does not establish train-time feasibility.

Synthetic self-target gaps were mechanics-only and must not be used for V0 hyperparameter selection.

## EXP-0020 — Public Stage-2 Teacher Asset — VERIFIED
Acquired only:
```text
NOVAglow646/Monet-SFT-7B/stage2
local: models/Monet-SFT-7B-stage2
```
Stage-1 and Stage-3 were not downloaded at that point.

Correct safetensors verification:
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
The Stage-2 asset is structurally verified and must not be re-downloaded.

## EXP-0021 — Real Stage-2 Teacher Gap Probe — FAILED PRE-SPECIFIED VIABILITY GATE
Frozen metric-calibration cohort:
```text
seed: 20260914
rows: [48724,53111,59512,68981,81393,90224,94458,117170]
selection SHA256: 57472e4bfd0e7de389e17a56bfbcfb31f8efdc0c2dfbdb8b07d3c03156d93428
```
Teacher: verified public Stage-2 checkpoint. Student probe: existing final `models/Monet-7B`. Real teacher/student tensor shape: `[29,8,3584]`.

Pre-specified preferred metric:
```text
D_hidden = mean_{layer,latent} [1 - cosine(T[l,t,:], S[l,t,:])]
gap = D_hidden(negative) - D_hidden(positive)
```
Frozen viability gate:
```text
positive gaps >= 6/8
AND median gap > 0
```
Observed explicit-`dim=-1` result:
```text
positive gaps: 5/8
mean gap:   +0.0012064651
median gap: +0.0022175312
min gap:    -0.0057374239
max gap:    +0.0071036220
one-sided sign-test p: 0.36328125
preferred_metric_viability_gate_passed: false
```
Official default-`dim=1` control was weaker:
```text
positive gaps: 3/8
median gap: -0.0002297163
one-sided sign-test p: 0.85546875
```
Do not declare either metric validated. Do not switch to the official metric, filter failed rows, alter the frozen cohort, or relax the gate.

Important failure structure: rows `53111` and `68981` had reversed explicit-`dim=-1` gaps across all `29/29` layers; `117170` was also negative in aggregate. This is not merely a threshold-edge failure.

## Candidate V0 Loss — NOT FROZEN
Keep the matched baseline exactly:
```text
L_base = L_CE + lambda_align * L_align_official
```
The teacher-anchored explicit-`dim=-1` ranking metric is **not validated** by EXP-0021 on final `Monet-7B` and cannot yet be frozen as `L_evidence`.

However, EXP-0021 used final `Monet-7B` as the student probe checkpoint. Official Stage-3 training initializes the student from the public Stage-1 checkpoint, not from final Monet. The Step-28 protocol explicitly limited this run to metric direction/scale under that checkpoint mismatch. Therefore one clean discriminator remains before rejecting the metric family: rerun the exact same frozen 8-row cohort, metric, negative operator, and viability gate with the official Stage-1 checkpoint as the student.

## Active Constraints
- No architecture change.
- V0 is SFT-only; no VLPO changes yet.
- Visual_CoT only for the first V0 operator.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 negative operator unchanged.
- Preserve official Stage-3 baseline alignment exactly in matched controls.
- Do not metric-shop after EXP-0021.
- Do not change the frozen 8-row EXP-0021 cohort or viability gate for the Stage-1 rerun.
- Do not re-download Stage-2.
- Stage-3 checkpoint remains unnecessary for shape/mechanics inspection.
- Any continued-SFT V0 experiment must have a matched continued-SFT control with identical data/steps except the new evidence loss.

## Decision Rule for the Next Probe
Acquire only the public Stage-1 checkpoint used to initialize official Stage-3 training, then rerun the **same** EXP-0021 real-teacher probe.

If Stage-1 passes the unchanged gate, the checkpoint mismatch explains why final-Monet calibration was not representative and the metric family can proceed to gradient/loss-form testing.

If Stage-1 also fails the unchanged gate, reject teacher-anchored explicit-`dim=-1` natural ranking as the first V0 evidence metric. Do not rescue it by changing rows, thresholds, layers, or post-hoc filters.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Validate/freeze the Visual_CoT full-image neutral-occlusion operator.
- [x] Audit static Stage-3 loss path.
- [x] Establish runtime tensor contract.
- [x] Acquire and structurally verify Stage-2 teacher.
- [x] Run real-teacher metric-calibration cohort on final Monet; frozen gate failed.
- [ ] Acquire only public Stage-1 checkpoint used by official Stage-3 initialization.
- [ ] Rerun the exact frozen 8-row real-teacher probe with Stage-1 student.
- [ ] Decide whether to keep or reject the teacher-anchored metric family.
- [ ] Only then freeze `L_evidence` and run a deterministic training smoke test.

## Next Action
Acquire and structurally verify only `NOVAglow646/Monet-SFT-7B/stage1`, pinned to its public Stage-1 upload commit. Do not download Stage-3 and do not modify the EXP-0021 cohort/metric/gate.