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
The pre-specified same-teacher candidate failed the frozen gate on both final Monet and official Stage-1 student initialization. Therefore reject:
```text
D(T, S_negative) > D(T, S_positive)
```
for one shared helper-derived Stage-2 target. Do not rescue it by changing rows, layers, latent positions, thresholds, filters, margins, or switching post-hoc to official `dim=1`.

## EXP-0024 — Fresh Counterfactual-Delta Cohort Freeze — PASSED
Fresh outcome-blind cohort:
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
Mechanical result: `12/12` valid.

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
Frozen gate:
```text
all 12 runtime samples valid and finite
AND positive primary scores >= 10/12
AND median primary score > 0
AND mean primary score > 0
```
Observed:
```text
runtime valid: 12/12
positive primary scores: 10/12
mean primary score: +0.0067885655
median primary score: +0.0069745332
one-sided exact sign-test p: 0.019287109375
gate: true
```

## D015 Evidence Metric — FROZEN; WEIGHT NOT YET FROZEN
Direct evidence loss:
```text
L_evidence_raw = 1 - score
```
Frozen gradient policy:
```text
T+, T-, Delta_T, w: stop-gradient
S+: differentiable
S-: differentiable
```
No margin, temperature, layer selector, latent-position selector, or secondary metric substitution.

`lambda_evidence` remains unfrozen.

## EXP-0026 — Representation-Gradient Audit — PASSED
The exact D015 loss was reconstructed on the same frozen 12-row cohort with zero score-reconstruction error.

Mechanical contract:
```text
all_12_gradient_contracts_passed: true
teacher_and_weights_stop_gradient: true
both_student_branches_differentiable: true
max_score_reconstruction_error: 0.0
```
Representation-gradient magnitudes:
```text
positive branch grad L2 mean:   0.0003022579282
positive branch grad L2 median: 0.0002328506162
negative branch grad L2 mean:   0.0003041839409
negative branch grad L2 median: 0.0002404905899
```
Positive-vs-negative flattened gradient cosine:
```text
mean:   -0.9006003042
median: -0.8951198161
```
The near-opposite branch gradients are mechanically consistent with optimizing a paired difference `Delta_S=N(S+)-N(S-)`; they are descriptive only and must not be used to set `lambda_evidence`.

This pass establishes representation-level differentiability only. It does not establish full-model parameter-gradient scale, ZeRO-2 training memory, optimizer stability, or benchmark improvement.

## Active Constraints
- No architecture change.
- V0 remains SFT-only; no VLPO changes yet.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 student negative operator unchanged.
- Preserve official Stage-3 baseline loss exactly in matched controls.
- Same-teacher ranking remains rejected.
- Do not change the fresh 12-row counterfactual calibration cohort to retune the metric.
- Do not select layers/latents or switch to a secondary diagnostic post-hoc.
- Do not choose `lambda_evidence` from raw loss magnitude or representation-gradient magnitude.
- Any eventual V0 run requires a matched control with identical data/steps except the evidence term.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Freeze Visual_CoT full-image neutral occlusion.
- [x] Resolve Stage-3 tensor/runtime contract.
- [x] Verify Stage-2 teacher and Stage-1 initialization.
- [x] Reject same-teacher natural ranking after two unchanged frozen-gate tests.
- [x] Freeze fresh counterfactual-delta cohort and paired teacher intervention.
- [x] Pass the pre-specified counterfactual-delta metric gate.
- [x] Verify finite/nonzero representation gradients to both student branches.
- [ ] Run model-level Stage-3 backward/memory smoke under official-style ZeRO-2.
- [ ] Measure model-parameter gradient scales for baseline vs raw evidence term.
- [ ] Freeze `lambda_evidence` using a pre-specified parameter-gradient criterion.
- [ ] Run a deterministic matched baseline vs V0 smoke test.

## Next Action
Run a model-level engineering smoke test with verified Stage-1 initialization under the official-style ZeRO-2 distributed path. Use one fixed engineering sample only to test backward/optimizer/memory feasibility and the exact two-branch evidence graph. Do not freeze `lambda_evidence` from that single sample. Parameter-gradient calibration must be a separate pre-specified step after memory feasibility is established.