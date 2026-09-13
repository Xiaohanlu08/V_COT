# Current State

## Project Stage
V0 preparation: evidence-supported latent contrastive supervision.

## Current Objective
Translate the now-confirmed target-specific latent sensitivity signal into the minimal V0 SFT intervention defined in `PROJECT_GOAL.md`, without changing Monet's architecture or RL pipeline.

## Pinned Baseline
- Monet: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Checkpoint: local `models/Monet-7B`
- VLMEvalKit snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Protected runtime unchanged: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, transformers 4.54.0, trl 0.15.2, vllm 0.10.0.

## Verified Results So Far
- Natural VStarBench latent triggering: `72/191 = 37.7%`; all natural segments length 10.
- Paired latent-start suppression over the 72 naturally-triggered examples: after option-aware rescoring, baseline and latent-off are both `54/72 = 0.75`; McNemar exact `p=1.0`. No answer-level utility effect detected.
- Exact recurrent latent tensor capture is verified: observation-only instrumentation captures the `10 x 3584` recurrent `st["pending"]` states without changing baseline generation.
- Fixed-prefix/fixed-trigger replay is mechanically valid. On development position 0, original-image replay vs the natural reference gives mean cosine `0.9999062`, min cosine `0.9998059`, mean relative L2 `0.0133799`.
- Crop-based positive-vs-negative evidence comparison on position 0 was rejected as geometry-confounded (`Delta_evidence=-0.0050027`).
- Full-image target-vs-sham occlusion on development position 0 produced a promising target-specific signal and motivated confirmatory replication.
- Triggered-72 annotation audit mapped `69/72`; all 41 mapped `direct_attributes` cases have exactly one target and one bbox.

## EXP-0010 — Outcome-Blind 12-Sample Cohort Freeze — COMPLETE
Frozen confirmatory positions:
```text
[6, 11, 22, 23, 27, 53, 57, 59, 67, 71, 102, 112]
```
Frozen cohort SHA-256:
`f65ebdbefc8a73276877f4096d5cb8897339447c91a31dc68322a811dda487e4`

The cohort was selected before any new target-occlusion outcomes were observed. Development position 0 was excluded.

## EXP-0011 — Confirmatory 12-Sample Target-vs-Sham Occlusion — PRIMARY GATE PASSED
Pre-outcome protocol: `protocols/CONFIRMATORY_OCCLUSION_12.md`.

Per sample, the exact natural pre-latent prefix was replayed, the original image was required to naturally predict `151666=<abs_vis_token>` at the fixed boundary before override, and aligned `10 x 3584` recurrent states were compared under one target mask and 32 deterministic same-size sham masks.

Mechanical validity:
```text
mechanically_valid_samples: 12
mechanically_invalid_samples: 0
all_12_mechanically_valid: true
```

Cross-sample results:
```text
positive specificity: 10/12
negative specificity: 2/12
specificity mean:   +0.0008347084
specificity median: +0.0004564524
target percentile mean:   83.8542
target percentile median: 100.0
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p:      0.0009765625
primary_hypothesis_supported: true
```

Frozen primary rule:
```text
all 12 mechanically valid
AND median specificity > 0
AND exact one-sided sign-test p < 0.05
```
All three conditions were satisfied. The result is therefore a confirmed multi-sample signal, under the pre-specified direct-attribute/full-image-neutral-occlusion protocol, that masking task-relevant target evidence perturbs Monet's aligned recurrent latent trajectory more than matched nuisance occlusions.

The result must still **not** be described as proof that latent reasoning is useful for answers or as universal visual grounding. The earlier latent-off experiment found no answer-level correctness effect. What is now supported is the narrower causal premise required for V0: recurrent latent states are selectively sensitive to task-relevant visual evidence on this direct-attribute protocol.

## V0 Gate Decision
The pre-specified evidence gate for beginning V0 is **OPEN**.

This means it is now justified to implement the minimal SFT-only experiment in `PROJECT_GOAL.md`:
1. construct evidence-preserving and evidence-destroying visual views;
2. extract corresponding latent states;
3. add latent contrastive/evidence supervision;
4. keep architecture and RL unchanged;
5. evaluate against a matched reproduced Monet baseline before any move to V1/V2.

The confirmatory VStarBench cohort remains evaluation/probing evidence and must not be used as V0 training data.

## Monet V0 Integration Facts — UPSTREAM VERIFIED
Official Monet Stage 3:
- precomputes teacher latent embeddings with `src.precompute_teacher_latents`;
- trains through `src.main` with `CustomTrainerSFT_STAGE3`;
- `CustomTrainerSFT_STAGE3` first performs a latent forward to obtain `ce_patch_vec`, then a CE/alignment forward;
- Stage-3 student image tensors are built in `collate_fn_sft_stage3` from user/question images only;
- the official Stage-3 objective is `student_ce_loss + alignment_weight * alignment_loss`.

These are the minimal hook points for V0. V0 should extend Stage 3 rather than create a separate architecture.

## Formal Evaluation Caveat
Monet's README requests a supplementary API judge but does not identify the exact judge model/configuration. Local option-aware rescoring is deterministic but is not that under-specified API judge. This does not block V0 implementation, but final benchmark claims must use a fixed documented evaluator.

## Next Milestones
- [x] Pin Monet / runtime / benchmark path.
- [x] Characterize natural latent triggering over all VStarBench samples.
- [x] Validate exact latent-start suppression and robust local answer rescoring.
- [x] Capture exact recurrent latent hidden tensors without changing generation.
- [x] Validate fixed-prefix/fixed-trigger replay.
- [x] Reject crop-based evidence comparison as geometry-confounded on sample 0.
- [x] Obtain promising development-sample full-image target-vs-sham specificity.
- [x] Audit official annotations and freeze an outcome-blind confirmatory cohort.
- [x] Pass the pre-specified 12-sample target-specificity confirmatory gate.
- [ ] Audit local Monet SFT Stage-3 assets/checkpoints/dataset availability before modifying training code.
- [ ] Implement V0 as a minimal Stage-3 extension with evidence-aware latent contrastive loss.
- [ ] Run a tiny deterministic V0 smoke test before any multi-GPU training.
- [ ] Reproduce a matched Monet training/evaluation baseline under the same local setup.
- [ ] Run V0 pilot and compare against the matched baseline.
- [ ] Move to V1/V2 only if V0 improves the predefined aggregate benchmark metric.

## Next Action
Do not run more VStarBench occlusion probes and do not alter the confirmatory cohort. The next task is a no-training audit of local Monet Stage-3 training assets and exact source hook points, followed by a tiny V0 implementation/smoke test. Do not start full V0 training yet.
