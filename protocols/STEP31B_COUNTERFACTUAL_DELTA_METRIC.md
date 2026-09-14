# Step 31b — Counterfactual-Delta Metric Validation

## Frozen cohort
Use exactly the EXP-0024 rows:
```text
[5345,43387,48794,56294,64481,69518,87598,92915,94395,103479,111682,113005]
```
Selection SHA-256:
```text
bdd8e027b8f9aa367045dcda049d12aa4745bb3c4f918cc0b6433c7172826798
```
Do not change rows after seeing outcomes.

## Models and views
Teacher model: verified public Stage-2.

Student model: verified public Stage-1, the official Stage-3 initialization.

Teacher positive:
```text
T+ = Stage-2 all-layer latent tensor from original source + official helper
```

Teacher negative:
```text
T- = Stage-2 all-layer latent tensor from frozen Step-24 occluded source + same-size neutral helper constructed in Step 31a
```

Student positive:
```text
S+ = Stage-1 Stage-3-path aligned hidden tensor from original full source
```

Student negative:
```text
S- = Stage-1 Stage-3-path aligned hidden tensor from frozen Step-24 occluded source
```

Expected tensor shape for all four conditions:
```text
[29,8,3584]
```

## Pre-specified primary metric
First normalize each hidden vector independently along the hidden dimension:
```text
N(X)[l,t,:] = X[l,t,:] / ||X[l,t,:]||_2
```

Define counterfactual deltas:
```text
Delta_T = N(T+) - N(T-)
Delta_S = N(S+) - N(S-)
```

For each layer/latent cell:
```text
c[l,t] = cosine(Delta_T[l,t,:], Delta_S[l,t,:], dim=-1)
w[l,t] = ||Delta_T[l,t,:]||_2
```

Primary sample score:
```text
score = sum(w[l,t] * c[l,t]) / sum(w[l,t])
```
Teacher-delta magnitude is used only as a fixed relevance weight so cells with stronger teacher counterfactual change contribute more. No layer or latent-position selection is allowed.

Desired direction:
```text
score > 0
```

## Secondary diagnostics — not eligible to replace the primary metric
Log only:
- unweighted mean of `c[l,t]`;
- mean and median teacher delta norm;
- mean and median student delta norm;
- per-layer weighted/unweighted scores;
- per-latent weighted/unweighted scores.

These diagnostics may explain failure but cannot be substituted post-hoc as the primary metric.

## Frozen viability gate
All 12 runtime samples must be mechanically valid and finite, and:
```text
positive primary scores >= 10/12
AND median primary score > 0
AND mean primary score > 0
```
`10/12` corresponds to one-sided exact sign-test p = 0.019287109375 under p=0.5.

## Interpretation
PASS licenses the counterfactual-delta metric family for a later gradient/loss-form smoke test. It does not demonstrate benchmark improvement and does not freeze the final loss weight or gradient policy.

FAIL rejects this normalized teacher-weighted delta-direction metric as the first replacement evidence metric. Do not rescue it by selecting layers/latents, filtering samples, changing weights, or switching to a secondary diagnostic on this cohort.

## Baseline constraint
The official Stage-3 objective remains unchanged in matched controls:
```text
L_base = L_CE + lambda_align * L_align_official
```
Any future counterfactual-delta term is additional and must not silently alter the baseline alignment implementation.