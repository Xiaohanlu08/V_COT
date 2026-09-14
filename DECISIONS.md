# Decision Log

This file records project-level decisions so that rejected ideas are not accidentally reintroduced later without new evidence.

## D001 — Use Monet as the initial framework
**Status:** ACTIVE

**Decision:** Use Monet with Qwen2.5-VL-7B as the initial open-source latent visual reasoning framework.

**Reason:** The project requires an existing continuous visual latent reasoning pipeline that can be reproduced before introducing new supervision. Monet is the selected starting point for the current project plan.

**Revisit condition:** Reopen only if the Monet baseline cannot be reproduced reliably, the required code path is unavailable, or a clearly superior and equally reproducible open-source base is identified.

---

## D002 — Baseline reproduction comes before method modification
**Status:** ACTIVE

**Decision:** Do not modify the architecture, latent loss, data pipeline, or VLPO logic until the selected Monet baseline is reproduced and frozen.

**Reason:** Without a matched reproduced baseline, later gains or regressions cannot be attributed reliably to the proposed method.

**Revisit condition:** None. This is a project-control requirement.

---

## D003 — Validate the hypothesis with SFT before touching VLPO
**Status:** ACTIVE

**Decision:** V0 will test latent visual-evidence supervision in the supervised-training path only. VLPO/on-policy optimization is deferred to V2.

**Reason:** This isolates the scientific hypothesis from RL engineering complexity and provides an early stop condition if the idea does not improve the baseline.

**Revisit condition:** Reopen only after V0 evidence exists.

---

## D004 — Do not pursue text-only CoT optimization
**Status:** REJECTED FOR THIS PROJECT

**Idea:** Improve performance primarily through textual Chain-of-Thought prompting or textual reasoning supervision.

**Reason:** The core research question concerns continuous visual latent reasoning and visual evidence grounding, not text-only reasoning.

**Revisit condition:** Only if used as a controlled baseline or ablation, not as the main method.

---

## D005 — Do not add handcrafted visual experts by default
**Status:** REJECTED AS DEFAULT DIRECTION

**Idea:** Add another fixed segmentation/depth/edge/object expert simply to increase the number of visual token types.

**Reason:** The current hypothesis concerns identifying whether latent states are genuinely supported by useful visual evidence. Adding expert types does not directly solve latent-state usefulness or credit assignment.

**Revisit condition:** Only if an experiment demonstrates a specific missing perceptual capability that cannot be addressed by the evidence-gating formulation.

---

## D006 — Current target formulation
**Status:** PROVISIONAL / TO BE TESTED

**Decision:** The current target direction is Visual-Evidence-Gated Latent Reasoning, developed in three stages:

- V0: positive/negative visual views + latent-state extraction + latent contrastive supervision;
- V1: evidence-based gating/selection of latent supervision;
- V2: evidence-aware latent credit assignment integrated with Monet VLPO.

**Important:** This is a hypothesis under test, not a guaranteed improvement. Failure of V0 is sufficient reason to stop or redesign the method instead of forcing progression to V1/V2.

---

## D007 — Pin the Monet upstream baseline
**Status:** ACTIVE

**Decision:** Use `NOVAglow646/Monet` at commit `08939998d3d643a73a316e349faa34f420429153` as the immutable upstream baseline source for the first reproduction cycle. Use `NOVAglow646/Monet-7B` as the first official checkpoint to reproduce inference.

**Reason:** A fixed upstream SHA prevents later repository changes from silently altering the baseline and makes every subsequent patch attributable to a known source state.

**Revisit condition:** Only if the pinned commit is demonstrably broken for reproduction, or a later official commit fixes a blocking issue. Any change requires a new numbered decision and must record the old and new SHAs.

---

## D008 — Use 3090 for development and reserve H200 for justified scale
**Status:** ACTIVE

**Decision:** Use the available RTX 3090 system for baseline inference, latent-state inspection, three-view analysis, and V0 pilot development. Reserve H200-class hardware for full-scale experiments, multi-seed validation, or VLPO/RL workloads when 3090 memory/runtime is no longer efficient.

**Reason:** The first research questions can be falsified or validated without paying the cost of large-scale H200 runs. The official Monet SFT recipe uses 8 GPUs and ZeRO-2, so adapted smaller-GPU training must be labeled as such.

**Revisit condition:** Move an earlier stage to H200 only after a measured memory/runtime blocker is recorded.

---

## D009 — Prefer mirrors for large external downloads
**Status:** ACTIVE

**Decision:** Prefer mainland-friendly mirrors for large external downloads when source identity can still be verified.

**Reason:** Server international bandwidth is limited. Mirror use must not weaken reproducibility: Git SHAs, repository identity, file structure, and published hashes remain authoritative.

**Revisit condition:** Change mirrors if availability or integrity becomes unreliable. Never commit credentials or tokens to the repository.

---

## D010 — Open the V0 implementation gate after confirmatory target-specific latent sensitivity
**Status:** ACTIVE

**Decision:** Begin implementation of the V0 SFT-only evidence-aware latent supervision path. Keep Monet architecture and VLPO/RL unchanged. V0 should be implemented as a minimal extension of the official Stage-3 supervised-training path, not as a new model architecture.

**Evidence:** The pre-specified 12-sample direct-attribute confirmatory protocol passed its frozen primary rule: 12/12 mechanically valid, 10/12 positive specificity, median specificity `+0.0004564524`, one-sided sign-test `p=0.0192871`, sign-flip mean `p=0.0009765625`.

**What this establishes:** Under the tested full-image neutral-occlusion protocol, Monet recurrent latent states are selectively more sensitive to task-relevant target evidence than to matched nuisance occlusions across the frozen direct-attribute cohort.

**What this does not establish:** It does not prove that latent states improve answer correctness or establish universal grounding.

**V0 constraint:** The confirmatory VStarBench cohort is evaluation/probing evidence only and must not be used as V0 training data.

---

## D011 — Use Visual_CoT as the first V0 evidence family and same-source matched shams as the candidate negative principle
**Status:** SUPERSEDED BY D012

**Decision:** Restrict the first V0 pilot to `Visual_CoT` and initially test same-source geometry-matched sham crops.

**Superseded because:** Step 23 failed its pre-specified constructibility gate with only `44/61` valid cases.

---

## D012 — Reject universal disjoint sham crops; validate full-image recovered-evidence neutral occlusion
**Status:** SUPERSEDED BY D013 AFTER SUCCESSFUL VALIDATION

**Decision:** Reject the universal disjoint-sham crop and test full-image neutral occlusion of the recovered evidence region.

**Reason:** It preserves full-image geometry, remains same-sample, and is constructible even when evidence occupies a large fraction of the image.

---

## D013 — Freeze full-image neutral occlusion and preserve the official Stage-3 tensor space
**Status:** ACTIVE

**Decision:** Freeze the first V0 Visual_CoT student-view operator as:
```text
positive = original full source image
negative = same source with recovered evidence region neutral-filled by surrounding-ring mean RGB
```
Preserve the official Stage-3 all-layer teacher/student tensor space and official baseline loss in matched controls.

**Evidence:** Step 24 passed the frozen mechanical gate on all `61/61` strong samples. Runtime audits established aligned tensors `[29,8,3584]` and the exact official Stage-3 baseline path.

**Important caveat:** The official all-layer `alignment_loss` uses PyTorch cosine default `dim=1`; preserve it in the baseline but do not assume it is the correct evidence metric.

---

## D014 — Reject same-teacher natural ranking; redesign V0 around counterfactual change
**Status:** ACTIVE

**Decision:** Reject the first proposed evidence-metric assumption that both original and evidence-occluded student states should be ranked against one shared helper-derived Stage-2 target using explicit hidden-vector cosine distance:
```text
D(T, S_negative) > D(T, S_positive)
```
Do not use this same-teacher ranking as the first V0 evidence loss.

**Evidence:** The pre-specified metric failed the unchanged viability gate twice on the same frozen 8-row calibration cohort:

Final `Monet-7B` student:
```text
positive gaps: 5/8
median gap: +0.0022175312
gate: false
```

Official Stage-1 student initialization:
```text
positive gaps: 3/8
mean gap: -0.0029229820
median gap: -0.0134561360
one-sided sign-test p: 0.85546875
gate: false
```
The Stage-1 rerun changed only the student checkpoint, so checkpoint mismatch does not rescue the assumption. The official default-`dim=1` control also failed and must not be selected post-hoc.

**What is rejected:** The same-target directional ordering assumption and losses that depend on it.

**What is not rejected:** the frozen Visual_CoT neutral-occlusion operator, the VStar target-specific latent-sensitivity evidence, official Stage-3 teacher alignment as part of the baseline, or paired visual interventions for an auxiliary V0 objective.

**Redesign principle:** Model the counterfactual change induced by removing task-relevant visual evidence rather than forcing both views to be ordered around one teacher state.

**Anti-metric-shopping rule:** Do not reuse the failed 8-row cohort to choose the replacement metric. Freeze a fresh outcome-blind Visual_CoT cohort first, mechanically validate the paired teacher intervention, then pre-specify the delta metric/gate before running latent inference.

---

## D015 — Freeze the counterfactual-delta metric family for V0
**Status:** ACTIVE / LOSS WEIGHT PENDING ENGINEERING CALIBRATION

**Decision:** Use the counterfactual-delta metric that passed EXP-0025 as the first V0 evidence-metric family. For all-layer tensors `[29,8,3584]`:
```text
N(X)[l,t,:] = X[l,t,:] / ||X[l,t,:]||_2
Delta_T = N(T+) - N(T-)
Delta_S = N(S+) - N(S-)
c[l,t] = cosine(Delta_T[l,t,:], Delta_S[l,t,:], dim=-1)
w[l,t] = ||Delta_T[l,t,:]||_2
score = sum(w*c) / sum(w)
```
where `T+`/`T-` are paired Stage-2 teacher states and `S+`/`S-` are paired Stage-1/Stage-3 student states.

**Evidence:** On the fresh outcome-blind 12-row Step-31a cohort, the pre-specified gate passed exactly as frozen:
```text
runtime valid: 12/12
positive primary scores: 10/12
mean score: +0.0067885655
median score: +0.0069745332
one-sided exact sign-test p: 0.019287109375
gate: true
```

**Direct loss form:** Use the monotonic loss
```text
L_evidence_raw = 1 - score
```
without introducing a margin, temperature, layer selector, latent-position selector, or secondary metric.

**Gradient policy:** Teacher tensors, `Delta_T`, and `w` are fixed/stop-gradient. The validated student quantity is `Delta_S=N(S+)-N(S-)`; therefore both student branches remain differentiable by default. A stop-gradient on either student branch would change the validated optimization target and requires a new documented decision if ever introduced.

**Not yet frozen:** `lambda_evidence`. Do not choose it from the raw scalar loss magnitude. First verify representation gradients, then measure parameter-gradient scale and train-time memory in the actual Stage-3 path.

**What this pass establishes:** Directional alignment of evidence-removal-induced latent change between teacher and student under the frozen paired Visual_CoT intervention on the fresh calibration cohort.

**What this does not establish:** Benchmark improvement, answer-level utility, full-data coverage, train-time stability, or an optimal evidence-loss weight.

**Revisit condition:** Reopen the metric family only if the direct loss is mechanically non-differentiable/unstable in the real Stage-3 path or if a matched V0 experiment fails and a new independently motivated formulation is pre-specified.

---

## Decision Protocol
Any future change that alters the scientific question, base framework, major loss formulation, evaluation criterion, hardware policy, pinned upstream source, or stage order must add a new numbered decision here. Do not silently overwrite an older decision; append a new entry that supersedes it and explain why.