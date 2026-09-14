# EXP-0026 — Counterfactual-Delta Representation-Gradient Audit

## Status
PASSED.

## Purpose
Verify that the frozen D015 evidence loss is exactly reconstructible and supplies finite, nonzero gradients to both student counterfactual branches before attempting a full model-level backward/memory test.

## Frozen loss
The metric/loss family was not changed from EXP-0025:
```text
N(X)[l,t,:] = X[l,t,:] / ||X[l,t,:]||_2
Delta_T = N(T+) - N(T-)
Delta_S = N(S+) - N(S-)
w[l,t] = ||Delta_T[l,t,:]||_2
score = sum(w * cosine(Delta_T, Delta_S, dim=-1)) / sum(w)
L_evidence_raw = 1 - score
```

Gradient policy frozen before the audit:
```text
T+, T-: stop-gradient
Delta_T: stop-gradient
w: stop-gradient
S+: differentiable
S-: differentiable
```
No margin, temperature, layer selector, latent-position selector, or one-branch stop-gradient was introduced.

## Cohort
Exact frozen fresh counterfactual cohort from EXP-0024/EXP-0025:
```text
rows: [5345,43387,48794,56294,64481,69518,87598,92915,94395,103479,111682,113005]
selection SHA256: bdd8e027b8f9aa367045dcda049d12aa4745bb3c4f918cc0b6433c7172826798
```

Teacher: verified public Stage-2 checkpoint.
Student: verified public Stage-1 checkpoint under the Stage-3 student path.
Tensor shape: `[29,8,3584]`.

## Mechanical gradient contract
All 12 samples passed:
```text
all_12_gradient_contracts_passed: true
teacher_and_weights_stop_gradient: true
both_student_branches_differentiable: true
max_score_reconstruction_error: 0.0
```

Representation-gradient magnitudes:
```text
positive_branch_grad_l2_mean:   0.0003022579282
positive_branch_grad_l2_median: 0.0002328506162
negative_branch_grad_l2_mean:   0.0003041839409
negative_branch_grad_l2_median: 0.0002404905899
```

Flattened positive-vs-negative representation-gradient cosine:
```text
mean:   -0.9006003042
median: -0.8951198161
```
The strongly negative cosine is mechanically consistent with an objective defined through the paired difference `Delta_S=N(S+)-N(S-)`; it is descriptive and is not used to choose a loss weight.

The EXP-0025 metric result was reproduced exactly during this audit:
```text
positive primary scores: 10/12
mean primary score: +0.0067885655
median primary score: +0.0069745332
one-sided exact sign-test p: 0.019287109375
counterfactual_delta_viability_gate_passed: true
```

## Interpretation
This experiment establishes representation-level differentiability of the frozen direct counterfactual loss and confirms that both student branches receive finite, nonzero gradients while teacher quantities remain detached.

It does **not** establish:
- full-model parameter-gradient scale;
- ZeRO-2 train-time memory feasibility;
- optimizer-step stability;
- an appropriate `lambda_evidence`;
- benchmark improvement or answer-level utility.

## Next action
Run a model-level Stage-3 backward/memory smoke test under the official-style ZeRO-2 distributed setup. First test memory/mechanics only; do not choose `lambda_evidence` from this representation-level gradient magnitude.