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

## Decision Protocol
Any future change that alters the scientific question, base framework, major loss formulation, evaluation criterion, or stage order must add a new numbered decision here. Do not silently overwrite an older decision; append a new entry that supersedes it and explain why.
