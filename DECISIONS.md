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
**Status:** SUPERSEDED BY D012

**Decision:** Restrict the first V0 pilot to the `Visual_CoT` helper family rather than mixing all six Monet-SFT-125K transformation families. Use the official helper crop as positive evidence and validate a negative constructed from the same source image with matched crop geometry and no overlap with the positive-evidence region.

**Evidence:** `Visual_CoT` contributes `118561/125072 = 94.7942%` of the official SFT data. The frozen 64-sample robust crop-recoverability audit passed unchanged thresholds: `64/64` pairs valid, `61/64 = 0.953125` strong recoverable, median same-source NCC `0.9923841`, and median same-minus-wrong margin `0.5103442`.

**Reason:** This construction directly controlled sample identity and crop geometry.

**Superseded because:** Step 23 failed its pre-specified mechanical gate: only `44/61 = 0.7213114754` strong samples had at least eight valid same-size disjoint sham positions. Seventeen samples were structurally nonconstructible, mainly because the recovered evidence crop occupied too much of the source image.

---

## D012 — Reject universal disjoint sham crops; validate full-image recovered-evidence neutral occlusion as the V0 negative
**Status:** SUPERSEDED BY D013 AFTER SUCCESSFUL VALIDATION

**Decision:** Keep `Visual_CoT` as the first V0 evidence family, but reject the exact D011 same-source disjoint sham-crop operator as the universal first negative. The next V0 negative candidate is a full-image evidence-destroyed view: recover the task-relevant source region from the official helper crop, preserve the original source dimensions, and neutral-fill that recovered region using a local surrounding-ring RGB statistic.

**Evidence:** Step 22b validated `Visual_CoT` helper recoverability on a frozen 64-sample cohort (`61/64` strong, median same-source NCC `0.9923841`, median same-minus-wrong margin `0.5103442`). Step 23 then failed the frozen disjoint-sham gate with only `44/61` constructible samples despite zero geometry violations and zero pixel-identical negatives among constructible cases.

**Reason:** Full-image neutral occlusion is constructible even when the evidence region is large, preserves input geometry, remains same-sample, and directly avoids the crop/full-image geometry confound identified in EXP-0007. It also reuses the intervention family that previously produced target-specific latent sensitivity in the frozen VStarBench confirmatory experiments.

---

## D013 — Freeze full-image neutral occlusion for V0 and preserve the official Stage-3 teacher/student tensor space
**Status:** ACTIVE

**Decision:** Freeze the first V0 Visual_CoT data operator as:
```text
positive/student view = original full source image
negative/student view = same source image with the recovered evidence region neutral-filled by surrounding-ring mean RGB
teacher target = official helper-derived cached Stage-3 teacher representation
```
Do not return to the failed universal disjoint-sham crop for the first V0 pilot.

**Evidence for operator:** Step 24 passed its frozen mechanical gate on all 61 Step-22b strong examples: `61/61` valid, zero dimension/outside-region/ring/change failures, and `frozen_gate_passed=true`. The negative keeps the original full-image geometry and changes only the recovered evidence region.

**Alignment-space decision:** Preserve the same all-layer teacher/student tensors used by Monet's official Stage-3 alignment when designing the V0 evidence loss. Corrected Step 25b confirms that official teacher precompute uses Stage-2 `outputs.hidden_states` with `--output_hidden_states`; Stage 3 uses `--alignment_layer all_layers`; student recurrent `ce_patch_vec` values are injected into latent-pad positions; and the second forward gathers student all-layer hidden states at those positions.

**Exact reduction caveat:** In the pinned source, the active all-layer `alignment_loss` receives tensors documented as `[num_layers, num_align, hidden_dim]` but calls `torch.nn.functional.cosine_similarity(...)` without specifying `dim`; PyTorch therefore uses default `dim=1`, i.e. the alignment-position axis, then averages the result. Preserve this exact behavior in the matched baseline, but do not automatically assume that scalar is the best V0 evidence-ranking metric. Step 26 must expose the runtime tensors so an explicit `dim=-1` per-hidden-vector cosine can be compared as a candidate additional metric without altering the baseline loss.

**Important code finding:** `affine_subspace_alignment_loss` is defined in the pinned model source but has no official Stage-3 call site. It must not be treated as the official Stage-3 alignment objective merely because the helper exists.

**Loss constraint:** Do not yet freeze a raw-`ce_patch_vec` cosine/triplet objective or blindly rank the official scalar alignment loss. The first candidate evidence term should compare original and evidence-occluded branches using the official teacher/student tensor space. Exact metric, ranking form, stop-gradient choice, margin/temperature, reduction, and weight require a runtime shape/memory probe first.

**Asset constraint:** Use the existing local `models/Monet-7B` for the minimum architecture-identical runtime probe before downloading the 16.6 GB public SFT Stage-2 or Stage-3 checkpoint. Download additional published assets only if the local probe cannot establish the required runtime contract.

**Revisit condition:** Reopen the operator only if the matched V0 pilot fails and a new pre-specified operator is justified. Reopen the tensor-space choice only if runtime inspection demonstrates that the official Stage-3 tensors cannot support the evidence comparison safely or efficiently.

---

## Decision Protocol
Any future change that alters the scientific question, base framework, major loss formulation, evaluation criterion, hardware policy, pinned upstream source, or stage order must add a new numbered decision here. Do not silently overwrite an older decision; append a new entry that supersedes it and explain why.
