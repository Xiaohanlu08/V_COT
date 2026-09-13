# Current State

## Project Stage
Latent-state characterization / controlled visual-evidence intervention.

## Current Objective
Determine whether Monet's recurrent latent hidden states are selectively sensitive to task-relevant visual evidence under a mechanically aligned, outcome-blind multi-sample counterfactual protocol before starting V0 training.

## Pinned Baseline
- Monet: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Checkpoint: local `models/Monet-7B`
- VLMEvalKit snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Protected runtime unchanged: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, transformers 4.54.0, trl 0.15.2, vllm 0.10.0.

## Verified Results So Far
- Natural VStarBench latent triggering: `72/191 = 37.7%`; all natural segments length 10.
- Paired latent-start suppression over the 72 naturally-triggered examples: after option-aware rescoring, baseline and latent-off are both `54/72 = 0.75`; McNemar exact `p=1.0`. No answer-level utility effect detected.
- Exact recurrent latent tensor capture is verified: observation-only instrumentation captures the `10 x 3584` recurrent `st["pending"]` states without changing the baseline generation.
- Fixed-prefix/fixed-trigger replay is mechanically valid. Original-image replay vs the natural reference on development position 0 gives mean cosine `0.9999062`, min cosine `0.9998059`, mean relative L2 `0.0133799`.
- Crop-based positive-vs-negative evidence comparison on position 0 was not supported (`Delta_evidence=-0.0050027`) because crop/resize geometry dominated.
- Full-image target-vs-sham occlusion on position 0 produced a promising target-specific signal: target cosine distance `0.00104058`, median sham distance `0.00048357`, specificity `+0.00055701`, target percentile `96.875`, only `1/32` sham masks >= target, empirical one-sided `p=0.0606061`. This remains a development-sample result only.
- Triggered-72 annotation audit mapped `69/72`; all 41 mapped `direct_attributes` cases have exactly one target and one bbox. Three repeated generic color questions at positions `29,49,61` remain unresolved and are excluded rather than manually resolved after outcomes.

## EXP-0010 — Outcome-Blind 12-Sample Cohort Freeze — COMPLETE
Step 17 froze the confirmatory direct-attribute cohort before any additional target-occlusion outcomes were observed.

Selection rule:
- naturally triggered;
- `category=direct_attributes`;
- official annotation resolved;
- exactly one target and one bbox;
- valid local image/bbox metadata;
- baseline latent segment length 10;
- development position 0 excluded;
- fixed `random.Random(20260913)` selection over sorted eligible samples.

Audit result:
```text
mapped_rows_total: 72
eligible_candidate_count: 39
unresolved excluded: 3
relative_position excluded: 28
development position 0 excluded: 1
invalid image/bbox metadata excluded: 1
invalid latent segment excluded: 0
```

Frozen positions:
```text
[6, 11, 22, 23, 27, 53, 57, 59, 67, 71, 102, 112]
```

Frozen annotation files:
```text
direct_attributes/sa_9019.json
direct_attributes/sa_5413.json
direct_attributes/sa_10919.json
direct_attributes/sa_26968.json
direct_attributes/sa_23893.json
direct_attributes/sa_29509.json
direct_attributes/sa_28073.json
direct_attributes/sa_38195.json
direct_attributes/sa_30935.json
direct_attributes/sa_10701.json
direct_attributes/sa_86882.json
direct_attributes/sa_7683.json
```

Frozen cohort SHA-256:
`f65ebdbefc8a73276877f4096d5cb8897339447c91a31dc68322a811dda487e4`

The cohort must not be changed based on subsequent specificity outcomes.

## Step 18 — Confirmatory 12-Sample Target-vs-Sham Occlusion PREPARED
Pre-outcome protocol is frozen in:
`protocols/CONFIRMATORY_OCCLUSION_12.md`

Implemented:
- `scripts/18_make_confirmatory_runner.py`
- `scripts/18_vstar_confirmatory_occlusion12.py`
- `scripts/18_vstar_confirmatory_occlusion12.sh`

Per sample:
1. replay the exact natural pre-latent token prefix as `prompt_token_ids`;
2. verify that, on the original image, the pre-force greedy next token is already `151666=<abs_vis_token>`;
3. force only the first replay token to latent-start for all visual conditions;
4. capture aligned `10 x 3584` recurrent latent states;
5. compare full-image target occlusion against 32 same-size sham occlusions;
6. deterministic sham seed is `20260913 + dataset_position`;
7. per-sample specificity is `target_distance - median(sham_distance)`.

Primary confirmatory cross-sample rule, frozen before execution:
```text
all 12 samples mechanically valid
AND median specificity > 0
AND exact one-sided sign-test p < 0.05
```
With 12 non-tied samples, the sign test requires at least 10 positive specificity values. Secondary analysis is an exact `2^12` sign-flip test on mean specificity.

No sample may be replaced because of a weak or negative result. Mechanical failures are reported, not substituted.

## Formal Evaluation Caveat
Monet's README requests a supplementary API judge but does not identify the exact judge model/configuration. Local option-aware rescoring is deterministic but is not that under-specified API judge. This does not block latent-state evidence experiments.

## Next Milestones
- [x] Pin Monet / runtime / benchmark path.
- [x] Characterize natural latent triggering over all VStarBench samples.
- [x] Validate exact latent-start suppression and robust local answer rescoring.
- [x] Capture exact recurrent latent hidden tensors without changing generation.
- [x] Validate fixed-prefix/fixed-trigger replay.
- [x] Reject crop-based evidence comparison as geometry-confounded on sample 0.
- [x] Obtain promising full-image target-vs-sham specificity on sample 0.
- [x] Audit official annotation mapping for the 72 naturally-triggered samples.
- [x] Freeze the outcome-blind 12-sample direct-attribute confirmatory cohort.
- [ ] Run Step 18 confirmatory full-image target-vs-sham occlusion over the frozen cohort.
- [ ] Apply the pre-specified cross-sample sign test and exact sign-flip secondary test.
- [ ] Design a separate relation-aware intervention for relative-position samples if needed.
- [ ] Start V0 only if the confirmatory evidence gate is supported.

## Next Action
Run only Step 18 on the frozen cohort. Do not change the cohort, target operator, sham count, seed rule, or primary statistic after observing outcomes.
