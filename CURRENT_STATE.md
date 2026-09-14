# Current State

## Project Stage
V0 preparation: evidence-discriminative latent supervision.

## Current Objective
Implement the smallest architecture-preserving Stage-3 SFT experiment that tests whether task-relevant visual evidence makes Monet recurrent latent states more useful.

## Pinned Baseline and Verified Assets
- Monet source: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Final inference checkpoint: local `models/Monet-7B`
- Verified Stage-2 teacher: local `models/Monet-SFT-7B-stage2`
- Verified Stage-1 Stage-3 initialization: local `models/Monet-SFT-7B-stage1`
- VLMEvalKit: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Runtime is protected; do not modify package versions without a demonstrated blocker.

## Evidence Gate — PASSED
Frozen 12-sample VStarBench direct-attribute target-vs-sham confirmatory result:
```text
mechanically valid: 12/12
positive specificity: 10/12
specificity median: +0.0004564524
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p: 0.0009765625
```
This supports target-specific latent sensitivity under the tested full-image neutral-occlusion intervention. It does not establish answer-level utility; latent-start suppression remained null (`54/72` vs `54/72`, McNemar `p=1.0`).

## First V0 Training Family and Frozen Student Operator
The first V0 pilot remains restricted to `Visual_CoT`.

Frozen student views:
```text
positive: original full source image
negative: same full image with recovered evidence region neutral-filled by surrounding-ring mean RGB
```
The neutral-occlusion operator passed its frozen `61/61` mechanical gate. Same-size disjoint-sham crops remain rejected.

## Stage-3 Tensor Contract — RESOLVED
```text
latent_size: 8
ce_patch_vec[b]: Tensor(8,3584)
ce_patch_pos == student_alignment_poss
aligned student all-layer hidden tensor: [29,8,3584]
real Stage-2 teacher tensor: [29,8,3584]
```
Official matched-control objective remains:
```text
L_base = L_CE + lambda_align * L_align_official
```
The pinned all-layer baseline alignment uses PyTorch cosine default `dim=1`; preserve that exact behavior in matched controls.

## Verified Model Assets
- Stage-2 teacher: structurally verified; do not re-download.
- Stage-1 Stage-3 initialization: structurally verified and all four upstream shard SHA-256 values match; do not re-download.
- Stage-3 checkpoint remains unnecessary for current preparation work.

## EXP-0021 / EXP-0023 — SAME-TEACHER NATURAL RANKING REJECTED
Frozen same-teacher candidate:
```text
D_hidden = mean_{layer,latent}[1 - cosine(T[l,t,:], S[l,t,:])]
gap = D_hidden(negative) - D_hidden(positive)
gate = positive gaps >= 6/8 AND median gap > 0
```
Final-Monet student: `5/8` positive gaps, gate false.

Official Stage-1 student: `3/8` positive gaps, median `-0.0134561360`, gate false.

Therefore reject the assumption:
```text
D(T, S_negative) > D(T, S_positive)
```
for one shared helper-derived Stage-2 target. Do not rescue it by changing rows, layers, latent positions, thresholds, filters, margins, or switching post-hoc to official `dim=1`.

## EXP-0024 — Fresh Counterfactual-Delta Cohort Freeze — PASSED
Data-only Step 31a used no latent/model inference and excluded development row `0` plus all EXP-0021/EXP-0023 rows.

Frozen fresh cohort:
```text
seed: 20260914
rows: [5345,43387,48794,56294,64481,69518,87598,92915,94395,103479,111682,113005]
selection SHA256: bdd8e027b8f9aa367045dcda049d12aa4745bb3c4f918cc0b6433c7172826798
```
Paired Stage-2 teacher views:
```text
T+ input = original source + official helper
T- input = frozen Step-24 occluded source + same-size neutral helper
```
The neutral helper uses that sample's already-frozen Step-24 surrounding-ring mean RGB.

Mechanical result: `12/12` valid. The rows and paired teacher intervention are frozen.

## EXP-0025 — Counterfactual-Delta Metric — PASSED PRE-SPECIFIED GATE
Frozen primary metric on tensors `[29,8,3584]`:
```text
N(X)[l,t,:] = X[l,t,:] / ||X[l,t,:]||_2
Delta_T = N(T+) - N(T-)
Delta_S = N(S+) - N(S-)
c[l,t] = cosine(Delta_T[l,t,:], Delta_S[l,t,:], dim=-1)
w[l,t] = ||Delta_T[l,t,:]||_2
score = sum(w*c) / sum(w)
```
Teacher counterfactual magnitude is the only weighting term; no layer/latent selection is used.

Frozen gate:
```text
all 12 runtime samples valid and finite
AND positive primary scores >= 10/12
AND median primary score > 0
AND mean primary score > 0
```
Observed result:
```text
runtime valid: 12/12
positive primary scores: 10/12
negative-or-zero: 2/12
mean primary score: +0.0067885655
median primary score: +0.0069745332
min: -0.0266901013
max: +0.0355119444
one-sided exact sign-test p: 0.019287109375
counterfactual_delta_viability_gate_passed: true
```
Secondary unweighted score was positive in aggregate (`mean +0.0031231965`, `median +0.0030090895`) but remains descriptive only.

## Evidence Metric Decision — COUNTERFACTUAL-DELTA FAMILY FROZEN
The validated replacement metric family is the teacher-delta-magnitude-weighted cosine alignment between `Delta_T` and `Delta_S` defined above.

Direct candidate loss form:
```text
L_evidence_raw = 1 - score
```
The teacher tensors, `Delta_T`, and weights `w` are fixed / stop-gradient. The validated `Delta_S` definition uses both student branches, so both `S+` and `S-` should remain differentiable unless a future engineering blocker forces a separately documented redesign.

Do not introduce a margin, temperature, layer selection, latent-position selection, or secondary metric substitution at this stage.

`lambda_evidence` is **not yet frozen**. Train-time gradient scale and memory feasibility must be measured before choosing it.

## Active Constraints
- No architecture change.
- V0 remains SFT-only; no VLPO changes yet.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 student negative operator unchanged.
- Preserve official Stage-3 baseline loss exactly in matched controls.
- Same-teacher ranking remains rejected.
- Do not change the fresh 12-row counterfactual calibration cohort after seeing EXP-0025 outcomes.
- Do not select layers/latents or switch to a secondary diagnostic post-hoc.
- Any eventual V0 run requires a matched control with identical data/steps except the new evidence term.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Freeze Visual_CoT full-image neutral occlusion.
- [x] Resolve Stage-3 tensor/runtime contract.
- [x] Verify Stage-2 teacher and Stage-1 initialization.
- [x] Reject same-teacher natural ranking after two unchanged frozen-gate tests.
- [x] Freeze fresh counterfactual-delta calibration cohort and paired teacher intervention.
- [x] Pass the pre-specified counterfactual-delta metric gate.
- [ ] Verify the direct loss `1-score` has finite/nonzero gradients to both student branches at representation level.
- [ ] Measure full Stage-3 + evidence-loss train-time memory/gradient behavior.
- [ ] Freeze `lambda_evidence` using pre-specified gradient-scale criteria.
- [ ] Run a deterministic matched baseline vs V0 smoke test.

## Next Action
Run a representation-gradient audit of the frozen direct loss `L_evidence_raw = 1 - score` without changing the metric. Confirm finite/nonzero gradients reach both `S+` and `S-`, while teacher deltas/weights remain stop-gradient. Do not choose `lambda_evidence` from loss magnitude alone; parameter-gradient scale must be measured later.