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

## First V0 Training Family and Frozen Operator
The first V0 pilot remains restricted to `Visual_CoT`.

Frozen data operator for eligible Visual_CoT examples:
```text
positive/student view: original full source image
negative/student view: same full image with recovered evidence region neutral-filled by surrounding-ring mean RGB
```
The neutral-occlusion operator passed its frozen `61/61` mechanical gate. The same-size disjoint-sham crop failed its frozen constructibility gate and remains rejected.

## Stage-3 Tensor Contract — RESOLVED
Static and runtime audits established:
```text
latent_size: 8
ce_patch_vec[b]: Tensor(8,3584)
ce_patch_pos == student_alignment_poss
aligned student all-layer hidden tensor: [29,8,3584]
real Stage-2 teacher tensor: [29,8,3584]
```
Official Stage-3 baseline objective remains:
```text
L_base = L_CE + lambda_align * L_align_official
```
The pinned all-layer `alignment_loss` uses `cosine_similarity(...)` without an explicit dimension, so the official reduction uses PyTorch default `dim=1` on `[layers, latent_positions, hidden_dim]`. Preserve that exact behavior in matched baseline controls.

## Verified Model Assets
### EXP-0020 — Stage-2 teacher — VERIFIED
The local Stage-2 safetensors payload exactly matches the index (`16578684928` bytes), container structure is exact, tensor ranges do not overlap, and index/header names match.

### EXP-0022 — Stage-1 Stage-3 initialization — VERIFIED
Public Stage-1 checkpoint is locally verified with exact safetensors structure and all four upstream shard SHA-256 hashes matching. Do not re-download Stage-1 or Stage-2. Stage-3 checkpoint remains unnecessary for the current preparation work.

## EXP-0021 — Same-Teacher Natural Ranking on Final Monet — FAILED FROZEN GATE
Frozen calibration cohort:
```text
seed: 20260914
rows: [48724,53111,59512,68981,81393,90224,94458,117170]
selection SHA256: 57472e4bfd0e7de389e17a56bfbcfb31f8efdc0c2dfbdb8b07d3c03156d93428
```
Pre-specified metric:
```text
D_hidden = mean_{layer,latent}[1 - cosine(T[l,t,:], S[l,t,:])]
cosine dim = -1
gap = D_hidden(negative) - D_hidden(positive)
```
Frozen gate: `positive gaps >= 6/8 AND median gap > 0`.

Final-Monet student result:
```text
positive gaps: 5/8
median gap: +0.0022175312
one-sided sign-test p: 0.36328125
gate: false
```

## EXP-0023 — Same-Teacher Natural Ranking on Official Stage-1 — FAILED FROZEN GATE
Step 30 changed exactly one factor from EXP-0021: student checkpoint `Monet-7B -> Monet-SFT-7B-stage1`. Stage-2 teacher, 8-row cohort, selection SHA, Step-24 negative operator, explicit `dim=-1` metric, reduction, and gate were unchanged.

Observed explicit-`dim=-1` result:
```text
positive gaps: 3/8
negative-or-zero gaps: 5/8
mean gap: -0.0029229820
median gap: -0.0134561360
min gap: -0.0699743032
max gap: +0.0632694364
one-sided sign-test p: 0.85546875
preferred_metric_viability_gate_passed: false
```
Official default-`dim=1` control also failed directionally:
```text
positive gaps: 4/8
median gap: -0.0007473230
one-sided sign-test p: 0.63671875
```
`STAGE1_STUDENT_REAL_TEACHER_GAP_PROBE_PASS=True` is only an execution-completion marker; scientifically the pre-specified metric gate failed.

## Metric Decision — SAME-TEACHER NATURAL RANKING REJECTED
Checkpoint mismatch is not the explanation for EXP-0021. The unchanged Stage-1 discriminator failed more strongly. Therefore reject the first proposed V0 evidence metric:
```text
D(T, S_negative) > D(T, S_positive)
```
with one shared official Stage-2 teacher target and explicit hidden-vector cosine `dim=-1`.

Do not rescue this branch by:
- changing the frozen 8 rows;
- selecting layers or latent positions after inspecting results;
- relaxing the `6/8` gate;
- filtering negative-gap samples;
- switching post-hoc to official `dim=1`;
- tuning a margin on the failed cohort.

This failure does **not** invalidate the frozen visual intervention or the VStar evidence-sensitivity result. It specifically rejects the assumption that evidence-preserving student states should naturally be closer than evidence-destroyed states to one shared helper-derived Stage-2 target.

## V0 Redesign Principle — COUNTERFACTUAL DELTA, NOT SAME-TARGET RANKING
The next candidate should represent the *change caused by destroying visual evidence* rather than ranking both views against one teacher target.

Provisional paired formulation:
```text
T+ = Stage-2 teacher under evidence-preserving source/helper input
T- = Stage-2 teacher under matched evidence-destroyed source/helper input
S+ = Stage-1/Stage-3 student under original full source
S- = same student under frozen Step-24 occluded source
Delta_T = normalize(T+) - normalize(T-)
Delta_S = normalize(S+) - normalize(S-)
```
A future evidence term may align `Delta_S` with stop-gradient `Delta_T`, while the official Stage-3 CE + alignment loss remains untouched. This is only a redesign hypothesis; the exact delta metric/loss is not frozen yet.

## Active Constraints
- No architecture change.
- V0 remains SFT-only; no VLPO changes yet.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 student negative operator unchanged.
- Preserve official Stage-3 baseline loss exactly in matched controls.
- Do not reuse the EXP-0021/0023 8-row cohort to select a replacement metric.
- Any replacement metric must be pre-specified and tested on a fresh outcome-blind Visual_CoT calibration cohort.
- Any eventual V0 run requires a matched control with identical data/steps except the new evidence term.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Freeze Visual_CoT full-image neutral occlusion.
- [x] Resolve Stage-3 tensor/runtime contract.
- [x] Verify Stage-2 teacher and Stage-1 student initialization.
- [x] Reject same-teacher explicit-`dim=-1` natural ranking after two unchanged frozen-gate tests.
- [ ] Freeze a fresh outcome-blind Visual_CoT cohort disjoint from EXP-0021/0023.
- [ ] Mechanically validate paired evidence-destroyed Stage-2 teacher inputs without looking at delta outcomes.
- [ ] Pre-specify and test counterfactual teacher/student delta alignment on that fresh cohort.
- [ ] Only after a passed gate, freeze `L_evidence` and run a deterministic training smoke test.

## Next Action
Run a data-only Step 31a: from the Step-22b strong Visual_CoT pool, exclude development row 0 and all EXP-0021/0023 rows, deterministically freeze a fresh calibration cohort, and mechanically construct the paired evidence-destroyed teacher view. Do not run teacher/student latent inference in the same step.