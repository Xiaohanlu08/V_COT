# EXP-0023 — Stage-1 Student / Real Stage-2 Teacher Gap Discriminator

**Status:** COMPLETED — FAILED PRE-SPECIFIED VIABILITY GATE

**Date:** 2026-09-14

**Purpose:** Determine whether the failed same-teacher natural-ranking result from EXP-0021 was caused by using final `Monet-7B` rather than the actual official Stage-3 student initialization.

## Frozen design
Changed exactly one factor from EXP-0021:
```text
student checkpoint:
models/Monet-7B -> models/Monet-SFT-7B-stage1
```

Unchanged:
```text
teacher: models/Monet-SFT-7B-stage2
rows: [48724,53111,59512,68981,81393,90224,94458,117170]
selection SHA256: 57472e4bfd0e7de389e17a56bfbcfb31f8efdc0c2dfbdb8b07d3c03156d93428
negative operator: frozen Step-24 full-image neutral occlusion
metric: explicit dim=-1 cosine distance on [29,8,3584]
gap: D(teacher, negative) - D(teacher, positive)
gate: positive gaps >= 6/8 AND median gap > 0
```

## Results
Preferred explicit-`dim=-1` metric:
```text
positive_gap_count: 3/8
negative_or_zero_gap_count: 5/8
mean_gap_neg_minus_pos: -0.0029229819774627686
median_gap_neg_minus_pos: -0.013456135988235474
min_gap: -0.06997430324554443
max_gap: +0.06326943635940552
one_sided_sign_test_p: 0.85546875
preferred_metric_viability_gate_passed: false
```

Official default-`dim=1` control:
```text
positive_gap_count: 4/8
negative_or_zero_gap_count: 4/8
mean_gap_neg_minus_pos: +0.0010189712047576904
median_gap_neg_minus_pos: -0.0007473230361938477
min_gap: -0.0229567289352417
max_gap: +0.026301562786102295
one_sided_sign_test_p: 0.63671875
```

The script completion marker `STAGE1_STUDENT_REAL_TEACHER_GAP_PROBE_PASS=True` indicates successful execution only; the scientific viability gate failed.

## Conclusion
**REJECT** same-teacher explicit-`dim=-1` natural ranking as the first V0 evidence metric.

Checkpoint mismatch does not explain EXP-0021. The official Stage-1 initialization failed the unchanged gate more strongly than final Monet. Do not rescue this branch by changing rows, thresholds, layers, latent positions, sample filters, or by switching post-hoc to the official default-`dim=1` metric.

This result does not invalidate the frozen visual intervention or the prior VStar target-specific latent-sensitivity evidence. It rejects only the directional assumption that evidence-preserving student states should naturally be closer than evidence-destroyed student states to one shared helper-derived Stage-2 teacher target.

## Next action
Move to a fresh-cohort redesign that represents the counterfactual *change* induced by evidence removal rather than ranking both views against one target. Freeze the new cohort and mechanically validate paired teacher positive/negative inputs before defining or testing the replacement metric.