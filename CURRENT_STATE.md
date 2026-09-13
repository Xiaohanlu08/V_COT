# Current State

## Project Stage
Latent-state characterization / controlled visual-evidence intervention.

## Current Objective
Determine whether Monet's recurrent latent hidden states are selectively sensitive to task-relevant visual evidence under a mechanically aligned counterfactual protocol, before starting V0 training.

## Pinned Baseline
- Monet: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Checkpoint: local `models/Monet-7B`
- VLMEvalKit snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Protected runtime unchanged: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, transformers 4.54.0, trl 0.15.2, vllm 0.10.0.

## Verified Results So Far
- Natural VStarBench latent triggering: `72/191 = 37.7%`; all natural segments length 10.
- Paired latent-start suppression over the 72 naturally-triggered examples: after option-aware rescoring, baseline and latent-off are both `54/72 = 0.75`; McNemar exact `p=1.0`. No answer-level utility effect detected.
- Exact recurrent latent tensor capture is verified: observation-only instrumentation captures the `10 x 3584` recurrent `st["pending"]` states without changing the baseline generation.
- Fixed-prefix/fixed-trigger replay is mechanically valid. Original-image replay vs the natural reference gives mean cosine `0.9999062`, min cosine `0.9998059`, mean relative L2 `0.0133799`.
- Crop-based positive-vs-negative evidence comparison on position 0 was not supported (`Delta_evidence=-0.0050027`) because crop/resize geometry dominated.
- Full-image target-vs-sham occlusion on position 0 produced a promising target-specific signal: target cosine distance `0.00104058`, median sham distance `0.00048357`, specificity `+0.00055701`, target percentile `96.875`, only `1/32` sham masks >= target, empirical one-sided `p=0.0606061`. This is not benchmark-level significance.

## EXP-0009 — Triggered-72 Official Annotation Mapping Audit — COMPLETE
Step 16 mapped the 72 naturally-triggered VLMEvalKit samples to official `craigwu/vstar_bench` annotations using category + normalized question, with option-string disambiguation for repeated questions.

Result:
```text
triggered_total: 72
official_annotation_file_count: 191
mapped_count: 69
unresolved_count: 3
resolution_counts:
  unique_question_category: 62
  question_category_plus_options: 7
  unresolved: 3
mapped_category_counts:
  direct_attributes: 41
  relative_position: 28
single_target_single_bbox_count: 49
single_target_single_bbox_category_counts:
  direct_attributes: 41
  relative_position: 8
unresolved_positions: [29, 49, 61]
VSTAR_TRIGGERED72_ANNOTATION_AUDIT_PASS=True
```

The three unresolved direct-attribute questions are repeated generic color questions (`man's cap`, `plastic stool`, `dog`) and are excluded from the confirmatory replication cohort rather than manually resolved after seeing outcomes.

## Confirmatory Replication Design
The next replication cohort is intentionally restricted to **direct_attributes** rather than mixing in `relative_position`:
- direct-attribute questions have a clean single target-property interpretation and all 41 mapped cases have exactly one target and one bbox;
- relative-position questions are relational and often require multiple objects/context, so they need a separate relation-aware intervention design;
- VStarBench position 0 is excluded because it was the development sample used to iterate the intervention design.

## Step 17 — Outcome-Blind Cohort Freeze PREPARED
Implemented:
- `scripts/17_freeze_direct_attribute_replication_cohort.py`
- `scripts/17_freeze_direct_attribute_replication_cohort.sh`

Preregistered selection rule:
1. Start only from the Step-16 mapping file.
2. Require natural latent triggering, resolved official annotation, `category=direct_attributes`, exactly one target and one bbox, valid image/bbox metadata, and baseline latent segment length 10.
3. Exclude development sample position 0.
4. Select exactly 12 samples using fixed `random.Random(20260913)` sampling over the sorted eligible set.
5. Save the frozen cohort and a SHA-256 digest before any additional target-occlusion outcomes are observed.

This selection is outcome-blind with respect to target-vs-sham specificity.

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
- [ ] Run Step 17 and freeze the 12-sample direct-attribute replication cohort.
- [ ] Run the same full-image target-vs-sham intervention over the frozen cohort.
- [ ] Aggregate per-sample specificity with paired/nonparametric statistics.
- [ ] Design a separate relation-aware intervention for relative-position samples if needed.
- [ ] Start V0 only if multi-sample evidence supports the visual-evidence gate.

## Next Action
Run only Step 17. It performs no model loading and no GPU inference. Do not run additional target-occlusion inference until the 12-sample cohort and its SHA-256 are frozen.
