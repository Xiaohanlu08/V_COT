# EXP-0025 — Counterfactual-Delta Metric Probe

## Status
PASSED the pre-specified viability gate.

## Purpose
Validate a replacement V0 evidence metric after same-teacher natural ranking was rejected in EXP-0021 and EXP-0023.

## Frozen cohort
Step 31a froze a fresh outcome-blind Visual_CoT cohort before any counterfactual-delta outcomes were observed:
```text
seed: 20260914
rows: [5345,43387,48794,56294,64481,69518,87598,92915,94395,103479,111682,113005]
selection SHA256: bdd8e027b8f9aa367045dcda049d12aa4745bb3c4f918cc0b6433c7172826798
```
The cohort excluded development row 0 and all EXP-0021/EXP-0023 rows.

## Paired views
Teacher Stage-2:
```text
T+ = original source + official helper
T- = frozen Step-24 occluded source + same-size neutral helper
```
The neutral helper uses the already-frozen Step-24 surrounding-ring mean RGB for that sample.

Student Stage-1:
```text
S+ = original source
S- = frozen Step-24 occluded source
```

## Frozen primary metric
For tensors `[29,8,3584]`, normalize each hidden vector along `dim=-1`:
```text
N(X)[l,t,:] = X[l,t,:] / ||X[l,t,:]||_2
Delta_T = N(T+) - N(T-)
Delta_S = N(S+) - N(S-)
c[l,t] = cosine(Delta_T[l,t,:], Delta_S[l,t,:], dim=-1)
w[l,t] = ||Delta_T[l,t,:]||_2
score = sum(w*c) / sum(w)
```
Teacher counterfactual magnitude is the only weighting term. No layer or latent-position selection is used.

## Pre-specified viability gate
```text
all 12 runtime samples valid and finite
AND positive primary scores >= 10/12
AND median primary score > 0
AND mean primary score > 0
```

## Result
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
Secondary unweighted cosine remained positive in aggregate:
```text
mean: +0.0031231965
median: +0.0030090895
```
This secondary result is descriptive only and was not used to pass the gate.

## Interpretation
The fresh cohort supports the pre-specified hypothesis that the direction of latent change caused by evidence destruction is aligned between the Stage-2 teacher and the Stage-1 student more often than chance under the tested paired Visual_CoT intervention.

This does **not** establish benchmark improvement, answer-level utility, train-time stability, or optimal loss weighting. It licenses the counterfactual-delta metric family for gradient/loss-form and train-time-memory testing.

## Constraint after pass
Do not change the validated metric by post-hoc layer/latent selection or substitute a secondary metric. The direct candidate loss is the monotonic transform `1 - score`; teacher deltas and teacher weights remain fixed/stop-gradient. `lambda_evidence` is not yet frozen.