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

## Rejected Branch — SAME-TEACHER NATURAL RANKING
EXP-0021 and EXP-0023 rejected the assumption
```text
D(T, S_negative) > D(T, S_positive)
```
for one shared helper-derived Stage-2 target. Do not rescue it by changing rows, layers, latent positions, thresholds, filters, margins, or switching post-hoc to official `dim=1`.

## Counterfactual-Delta Redesign — PASSED
Fresh frozen cohort:
```text
seed: 20260914
rows: [5345,43387,48794,56294,64481,69518,87598,92915,94395,103479,111682,113005]
selection SHA256: bdd8e027b8f9aa367045dcda049d12aa4745bb3c4f918cc0b6433c7172826798
```

Paired teacher/student views:
```text
T+ = Stage-2 teacher on original source + official helper
T- = Stage-2 teacher on Step-24 occluded source + same-size neutral helper
S+ = Stage-1/Stage-3 student on original source
S- = Stage-1/Stage-3 student on frozen Step-24 occluded source
```

Frozen metric:
```text
N(X)[l,t,:] = X[l,t,:] / ||X[l,t,:]||_2
Delta_T = N(T+) - N(T-)
Delta_S = N(S+) - N(S-)
c[l,t] = cosine(Delta_T[l,t,:], Delta_S[l,t,:], dim=-1)
w[l,t] = ||Delta_T[l,t,:]||_2
score = sum(w*c) / sum(w)
L_evidence_raw = 1 - score
```

EXP-0025 passed its pre-specified gate:
```text
runtime valid: 12/12
positive primary scores: 10/12
mean score: +0.0067885655
median score: +0.0069745332
one-sided exact sign-test p: 0.019287109375
gate: true
```

## D015 Evidence Metric — FROZEN; WEIGHT NOT YET FROZEN
Frozen gradient policy:
```text
T+, T-, Delta_T, w: stop-gradient
S+: differentiable
S-: differentiable
```
No margin, temperature, layer selector, latent-position selector, or secondary metric substitution.

`lambda_evidence` remains unfrozen.

## EXP-0026 — Representation-Gradient Audit — PASSED
The exact D015 loss reconstructed the frozen score with zero error and passed all 12 representation-gradient contracts:
```text
all_12_gradient_contracts_passed: true
teacher_and_weights_stop_gradient: true
both_student_branches_differentiable: true
max_score_reconstruction_error: 0.0
```
Mean representation-gradient L2 norms were `3.0226e-4` for `S+` and `3.0418e-4` for `S-`. Their flattened gradient cosine averaged about `-0.9006`, mechanically consistent with a paired difference objective. These values must not be used to set `lambda_evidence`.

## EXP-0027 — Official-Style ZeRO-2 Model-Level Smoke — ENGINEERING BLOCKER
The fixed row-5345 8-GPU RTX-3090 ZeRO-2 smoke failed from genuine per-rank CUDA capacity exhaustion **before backward**.

Baseline:
```text
positive latent forward completed
OOM in positive second Stage-3 forward (pos_out)
request: 12 MiB
free: ~19 MiB
PyTorch allocated: ~23.09 GiB / 23.57 GiB device
```

V0:
```text
OOM earlier in negative latent forward (neg_latent)
request: 20 MiB
free: ~21 MiB
PyTorch allocated: ~23.09 GiB / 23.57 GiB device
```

Reserved-but-unallocated memory was only about 17–21 MiB, and `expandable_segments=True` was already enabled. This is therefore a real 24-GB capacity blocker, not primarily fragmentation.

Because the **baseline itself OOMs**, EXP-0027 is not evidence against D015 and does not change the scientific metric status.

## Active Constraints
- No architecture change.
- V0 remains SFT-only; no VLPO changes yet.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 student negative operator unchanged.
- Preserve the official Stage-3 baseline loss exactly in matched controls.
- Same-teacher ranking remains rejected.
- Do not retune the frozen counterfactual metric on its calibration cohort.
- Do not choose `lambda_evidence` from raw loss magnitude, representation-gradient magnitude, or the single row-5345 engineering smoke.
- Any eventual V0 run requires a matched control with identical model/data/objective except the evidence term.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Freeze Visual_CoT full-image neutral occlusion.
- [x] Resolve Stage-3 tensor/runtime contract.
- [x] Verify Stage-2 teacher and Stage-1 initialization.
- [x] Reject same-teacher natural ranking.
- [x] Freeze and pass the counterfactual-delta metric gate.
- [x] Verify finite/nonzero representation gradients to both student branches.
- [x] Identify official-style ZeRO-2 on 24-GB RTX 3090 as a model-level memory blocker.
- [ ] Test the exact same baseline/V0 computation under ZeRO-3 with no CPU offload.
- [ ] If model-level training fits, pre-specify and measure parameter-gradient scales.
- [ ] Freeze `lambda_evidence` only from a separate multi-sample parameter-gradient calibration.
- [ ] Run a deterministic matched baseline vs V0 smoke test.

## Next Action
Run Step 32b2: keep row 5345, Stage-1 checkpoint, 2000-token Stage-3 image budget, CE/alignment objective, D015 evidence graph, and all scientific inputs unchanged; change only DeepSpeed memory partitioning from ZeRO-2 to ZeRO-3 with no CPU offload. This is an engineering memory discriminator, not a new scientific experiment and not lambda calibration.