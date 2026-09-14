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
This supports target-specific latent sensitivity under the tested full-image occlusion protocol. It does not establish answer-level utility; latent-start suppression remained null (`54/72` vs `54/72`, McNemar `p=1.0`).

## First V0 Training Family and Frozen Operator
The first V0 pilot is restricted to `Visual_CoT`.

Frozen data operator for eligible Visual_CoT examples:
```text
positive/student view: original full source image
negative/student view: same full image with recovered evidence region neutral-filled by surrounding-ring mean RGB
teacher target: official helper-derived Stage-2 all-layer teacher representation used by Stage-3
```
The neutral-occlusion operator passed its frozen 61/61 mechanical gate. The same-size disjoint-sham crop failed its frozen constructibility gate and remains rejected.

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

## EXP-0020 — Stage-2 Teacher Asset — VERIFIED
Local Stage-2 safetensors are structurally verified:
```text
index/payload bytes: 16578684928
container bytes: 16578766160
header overhead: 81232
all shard container sizes exact: true
all tensor ranges non-overlapping: true
index/header tensor names match: true
```
Do not re-download Stage-2.

## EXP-0021 — Real Stage-2 Teacher Gap Probe on Final Monet — FAILED FROZEN GATE
Frozen metric-calibration cohort:
```text
seed: 20260914
rows: [48724,53111,59512,68981,81393,90224,94458,117170]
selection SHA256: 57472e4bfd0e7de389e17a56bfbcfb31f8efdc0c2dfbdb8b07d3c03156d93428
```
Pre-specified candidate metric:
```text
D_hidden = mean_{layer,latent}[1 - cosine(T[l,t,:], S[l,t,:])]
cosine dimension = -1
gap = D_hidden(negative) - D_hidden(positive)
```
Frozen viability gate:
```text
positive gaps >= 6/8
AND median gap > 0
```
Observed with final `Monet-7B` student:
```text
positive gaps: 5/8
mean gap: +0.0012064651
median gap: +0.0022175312
one-sided sign-test p: 0.36328125
preferred_metric_viability_gate_passed: false
```
Official default-`dim=1` control was weaker (`3/8` positive gaps; median negative). Do not metric-shop or relax the gate.

Rows `53111` and `68981` had reversed explicit-`dim=-1` gaps across all 29/29 layers; `117170` was negative in aggregate. The final-Monet result is therefore not a simple threshold-edge miss.

## EXP-0022 — Stage-1 Student Initialization Asset — VERIFIED
Acquired and verified the public Stage-1 checkpoint used to initialize official Stage-3 training:
```text
NOVAglow646/Monet-SFT-7B/stage1
local: models/Monet-SFT-7B-stage1
pinned upload commit: fb8f3da99888c271a1732fb1ad6594f93f82a652
```
Verification:
```text
payload_matches_index_total: true
all_shard_container_sizes_exact: true
all_shards_nonoverlapping: true
per_shard_index_names_match: true
global_index_names_match: true
all_upstream_sha256_match: true
```
Stage-3 checkpoint remains unnecessary and was not downloaded.

## Candidate V0 Loss — NOT FROZEN
Teacher-anchored explicit-`dim=-1` natural ranking is not validated by EXP-0021. One pre-specified checkpoint discriminator remains because EXP-0021 used final Monet rather than the actual official Stage-3 initialization.

## Frozen Decision Rule for Stage-1 Rerun
Change exactly one factor:
```text
student checkpoint:
final Monet-7B -> public Stage-1
```
Keep unchanged:
```text
Stage-2 teacher
8-row cohort and selection SHA
Step-24 evidence operator
explicit dim=-1 metric
all-layer reduction
positive-gap threshold >= 6/8
median-gap threshold > 0
```
If Stage-1 passes, checkpoint mismatch explains why EXP-0021 was not representative and the metric family may proceed to gradient/loss-form testing.

If Stage-1 fails the same gate, reject teacher-anchored explicit-`dim=-1` natural ranking as the first V0 evidence metric. Do not rescue it by changing rows, layers, thresholds, filters, or switching post-hoc to the official `dim=1` metric.

## Active Constraints
- No architecture change.
- V0 is SFT-only; no VLPO changes yet.
- VStar confirmatory samples remain probing/evaluation only, never training data.
- Keep the frozen Step-24 negative operator unchanged.
- Preserve official Stage-3 baseline alignment exactly in matched controls.
- Do not change the frozen EXP-0021 cohort/metric/gate for the Stage-1 rerun.
- Do not re-download Stage-1 or Stage-2; both are verified.
- Do not download Stage-3 for this discriminator.
- Any eventual continued-SFT V0 run requires a matched continued-SFT control with identical data/steps except the new evidence loss.

## Next Milestones
- [x] Confirm target-specific latent sensitivity.
- [x] Freeze Visual_CoT full-image neutral occlusion.
- [x] Resolve Stage-3 tensor/runtime contract.
- [x] Verify public Stage-2 teacher.
- [x] Run final-Monet real-teacher calibration; frozen gate failed.
- [x] Verify public Stage-1 student initialization.
- [ ] Rerun the exact frozen 8-row real-teacher probe with Stage-1 student.
- [ ] Keep or reject the teacher-anchored metric family using the unchanged gate.
- [ ] Only then freeze `L_evidence` and run a deterministic training smoke test.

## Next Action
Run Step 30: same Stage-2 teacher, same frozen 8 rows, same Step-24 negatives, same explicit-`dim=-1` metric and viability gate, changing only the student checkpoint to `models/Monet-SFT-7B-stage1`.