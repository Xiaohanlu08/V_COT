# EXP-0024 — Fresh Counterfactual-Delta Calibration Cohort Freeze

**Status:** COMPLETED — PASSED MECHANICAL GATE

## Purpose
Freeze a fresh outcome-blind Visual_CoT cohort for the counterfactual-delta redesign after EXP-0021/EXP-0023 rejected same-teacher natural ranking. No latent/model outcomes were inspected in this step.

## Selection
- source pool: exact Step-22b `strong_recoverable` rows with valid Step-24 negatives
- exclusions: development row `0` and every EXP-0021/EXP-0023 calibration row `[48724,53111,59512,68981,81393,90224,94458,117170]`
- seed: `20260914`
- N: `12`
- selected rows: `[5345,43387,48794,56294,64481,69518,87598,92915,94395,103479,111682,113005]`
- selection SHA-256: `bdd8e027b8f9aa367045dcda049d12aa4745bb3c4f918cc0b6433c7172826798`

## Paired Teacher Views
Positive teacher view:
```text
original source + official Visual_CoT helper
```
Negative teacher view:
```text
frozen Step-24 occluded source + same-size helper fully neutral-filled
```
The negative helper uses that sample's already-frozen Step-24 surrounding-ring mean RGB, preserving helper dimensions while removing helper evidence.

## Mechanical Results
All 12 rows were valid:
```text
mechanical_gate_passed: true
invalid_rows: []
helper dimensions preserved: 12/12
helper changed fraction: >= 0.9956 for all 12
helper mean absolute RGB difference: >= 20.18 for all 12
latent_inference_performed: false
metric_outcomes_seen: false
```

## Conclusion
KEEP. The fresh 12-row cohort and paired teacher intervention are frozen before counterfactual-delta inference. The cohort must not be changed after viewing Step-31b delta outcomes.

## Next Action
Pre-specify a single primary counterfactual-delta metric and frozen viability gate, then run Stage-2 teacher / Stage-1 student inference on these exact 12 rows.