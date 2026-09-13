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

**Decision:** Use the available 4 x RTX 3090 system for baseline inference, latent-state inspection, three-view analysis, and V0 pilot development. Reserve H200-class hardware for full-scale experiments, multi-seed validation, or VLPO/RL workloads when 3090 memory/runtime is no longer efficient.

**Reason:** The first research questions can be falsified or validated without paying the cost of large-scale H200 runs. The official Monet SFT recipe uses 8 GPUs and ZeRO-2, so a 4 x 3090 training recipe will be treated as an adapted development configuration rather than silently labeled as the official recipe.

**Revisit condition:** Move an earlier stage to H200 only after a measured memory/runtime blocker is recorded.

---

## D009 — Prefer mirrors for large external downloads
**Status:** ACTIVE

**Decision:** Prefer mainland-friendly mirrors for large package/model/source downloads when source identity can still be verified. Current defaults are TUNA for PyPI/Conda, `hf-mirror.net` through `HF_ENDPOINT` for Hugging Face assets, and a GitHub download proxy only as a transport fallback for the public Monet clone.

**Reason:** Server international bandwidth is limited. Mirror use must not weaken reproducibility: the Monet Git SHA and model repository identity remain authoritative.

**Revisit condition:** Change mirrors if availability or integrity becomes unreliable. Never commit credentials or tokens to the repository.

---

## D010 — Open the V0 implementation gate after confirmatory target-specific latent sensitivity
**Status:** ACTIVE

**Decision:** Begin implementation of the V0 SFT-only evidence-aware latent supervision path. Keep Monet architecture and VLPO/RL unchanged. V0 should be implemented as a minimal extension of the official Stage-3 supervised-training path, not as a new model architecture.

**Evidence:** The pre-specified 12-sample direct-attribute confirmatory protocol (`protocols/CONFIRMATORY_OCCLUSION_12.md`) passed its frozen primary rule. All 12 samples were mechanically valid; 10/12 had positive target specificity; median specificity was `+0.0004564524`; the exact one-sided sign-test p-value was `0.0192871`; the exact sign-flip mean p-value was `0.0009765625`.

**What this establishes:** Under the tested full-image neutral-occlusion protocol, Monet recurrent latent states are selectively more sensitive to task-relevant target evidence than to matched nuisance occlusions across the frozen direct-attribute cohort.

**What this does not establish:** It does not prove that latent states improve answer correctness, because the earlier latent-start suppression experiment showed no matched answer-level accuracy effect. It also does not establish universal grounding outside the tested task/operator.

**V0 constraint:** The confirmatory VStarBench cohort is evaluation/probing evidence only and must not be used as V0 training data. V0 must use training data independent of this held-out probing cohort.

**Revisit condition:** If V0 fails to improve the matched Monet baseline under the predefined aggregate evaluation, stop or redesign V0 before moving to V1/V2.

---

## D011 — Use Visual_CoT as the first V0 evidence family and same-source matched shams as the candidate negative principle
**Status:** ACTIVE / NEGATIVE OPERATOR PENDING STEP 23 MECHANICAL VALIDATION

**Decision:** Restrict the first V0 pilot to the `Visual_CoT` helper family rather than mixing all six Monet-SFT-125K transformation families. Use the official helper crop as positive evidence and validate a negative constructed from the same source image with matched crop geometry and no overlap with the positive-evidence region.

**Evidence:** `Visual_CoT` contributes `118561/125072 = 94.7942%` of the official SFT data. The frozen 64-sample robust crop-recoverability audit passed unchanged thresholds: `64/64` pairs valid, `61/64 = 0.953125` strong recoverable, median same-source NCC `0.9923841`, and median same-minus-wrong margin `0.5103442`.

**Reason:** This construction directly controls the two major confounds already observed in the project. A same-source negative avoids sample-identity discrimination from unrelated images, and a same-geometry crop avoids the full-image-vs-crop geometry confound seen in EXP-0007. The other Monet helper families contain annotation, highlighting, state-transition, or geometry-construction transforms and therefore require separate operators.

**Constraint:** The exact same-source sham rule is not frozen until Step 23 passes its pre-specified mechanical gate. Do not start V0 training before that validation.

**Revisit condition:** Reopen if Step 23 fails mechanically, if the resulting V0 pilot shows no matched improvement, or if a later validated operator can safely include additional helper families.

---

## Decision Protocol
Any future change that alters the scientific question, base framework, major loss formulation, evaluation criterion, hardware policy, pinned upstream source, or stage order must add a new numbered decision here. Do not silently overwrite an older decision; append a new entry that supersedes it and explain why.
