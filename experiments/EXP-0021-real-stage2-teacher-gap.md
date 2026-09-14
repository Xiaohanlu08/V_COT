# EXP-0021 — Real Stage-2 Teacher Gap Probe

**Status:** COMPLETED; PRE-SPECIFIED VIABILITY GATE FAILED

**Date:** 2026-09-14

**Purpose:** Measure whether the frozen Visual_CoT full-image evidence occlusion moves Stage-3 student all-layer latent-position hidden states farther from the real official Stage-2 teacher target under the pre-specified explicit hidden-vector cosine metric.

## Frozen cohort
- seed: `20260914`
- rows: `[48724,53111,59512,68981,81393,90224,94458,117170]`
- selection SHA-256: `57472e4bfd0e7de389e17a56bfbcfb31f8efdc0c2dfbdb8b07d3c03156d93428`
- row 0 excluded because it was repeatedly used for development probes.

## Assets
- teacher: structurally verified public `models/Monet-SFT-7B-stage2`
- student probe checkpoint: existing local final `models/Monet-7B`
- teacher target shape: `[29,8,3584]`
- positive student view: original full image
- negative student view: frozen Step-24 neutral-occluded full image

## Metrics
Primary pre-specified candidate:
```text
explicit dim=-1 cosine distance
= mean over layer and latent position of
  1 - cosine(teacher[l,t,:], student[l,t,:])
```
Gap:
```text
gap = D(teacher, negative) - D(teacher, positive)
```
Desired direction: `gap > 0`.

Frozen viability gate:
```text
positive gaps >= 6/8
AND
median gap > 0
```
Official Stage-3 default-`dim=1` alignment was logged only as a control diagnostic.

## Results
Per-row explicit-`dim=-1` gaps:
```text
48724   +0.003003
53111   -0.004337
59512   +0.001432
68981   -0.005737
81393   +0.004538
90224   +0.005876
94458   +0.007104
117170  -0.002226
```
Aggregate:
```text
positive explicit-dim=-1 gaps: 5/8
mean gap:   +0.0012064651
median gap: +0.0022175312
min gap:    -0.0057374239
max gap:    +0.0071036220
one-sided sign-test p: 0.36328125
preferred_metric_viability_gate_passed: false
```
Official default-`dim=1` control:
```text
positive gaps: 3/8
mean gap:   +0.0000345632
median gap: -0.0002297163
one-sided sign-test p: 0.85546875
```

## Interpretation
The pre-specified `dim=-1` teacher-anchored metric did **not** pass its viability gate on the existing final `Monet-7B` student probe checkpoint. Therefore it must not be declared validated or frozen from this experiment.

The failure is not simply marginal noise: two reversed samples (`53111`, `68981`) show negative gaps across all `29/29` layers, and `117170` is negative at the aggregate level as well. The official default-`dim=1` control is worse and does not rescue the result.

However, this experiment intentionally used the final public `Monet-7B` as the student probe checkpoint, whereas official Stage-3 training initializes the student from the public Stage-1 checkpoint. The protocol explicitly limited EXP-0021 to metric-direction/scale calibration under that checkpoint mismatch. Thus this failure does not yet establish that the same metric is invalid at the actual Stage-3 initialization.

## Conclusion
**INCONCLUSIVE FOR THE ACTUAL STAGE-3 INITIALIZATION; FAILED FOR FINAL-MONET PROBE.**

Do not change the cohort, metric, or viability gate. Acquire only the official public Stage-1 checkpoint and rerun the exact same frozen 8-row real-teacher probe with Stage-1 as the student. If the same gate fails at Stage-1, reject teacher-anchored explicit-`dim=-1` natural ranking as the first V0 evidence metric rather than metric-shopping or post-hoc filtering.
