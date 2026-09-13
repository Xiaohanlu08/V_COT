# Experiment Log

This file is the permanent record of experiments that were actually run. Planned experiments should stay in `CURRENT_STATE.md` until execution begins.

## EXP-0000 — Repository Initialization
**Status:** COMPLETED

Established `PROJECT_GOAL.md`, `CURRENT_STATE.md`, `DECISIONS.md`, and `EXPERIMENTS.md` before modifying model behavior.

---

## EXP-0001 — VStarBench single-sample natural latent-trigger probe
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Verify unforced natural latent activation on the real VStarBench VLMEvalKit path while observationally retaining raw vLLM token IDs.

**Settings:** `LATENT_SIZE=10`, greedy decoding, no forced tokens, `allowed_token_ids=None`.

**Result:** sample/index 0 naturally emitted one latent segment from generated-token position 25 to 35; `NATURAL_LATENT_TRIGGER=True`; final answer matched ground truth.

**Conclusion:** KEEP.

---

## EXP-0002 — VStarBench 20-sample natural latent-trigger pilot
**Status:** COMPLETED

**Date:** 2026-09-13

**Data:** 20 VStarBench samples selected without replacement with `VCOT_SEED=20260913`.

**Settings:** `LATENT_SIZE=10`, greedy decoding, no forced tokens.

**Result:** triggered `8/20=0.40`; all markers balanced; no multi-segment behavior; all latent segments length 10. Triggered examples were longer and less often correct under the diagnostic parser than non-triggered examples.

**Conclusion:** KEEP. Instrumentation clean; full-dataset characterization justified.

---

## EXP-0003 — Full VStarBench natural latent-trigger characterization
**Status:** COMPLETED

**Date:** 2026-09-13

**Scripts:** `scripts/09_vstar_natural_trigger_scan.py`, `scripts/09_vstar_natural_trigger_scan.sh`

**Data:** complete VStarBench, all 191 examples.

**Settings:** `LATENT_SIZE=10`, greedy decoding, no forced tokens, official Monet system prompt.

**Primary results:**
- triggered: `72/191 = 0.3769633508`
- balanced markers: 191/191
- multi-segment: 0/191
- all 72 latent segments length 10
- overall diagnostic option accuracy: `116/191 = 0.60733`

**Triggered vs non-triggered:** triggered `34/72=0.472222`, mean tokens `79.778`; non-triggered `82/119=0.689076`, mean tokens `45.773`.

**Exploratory tests:** Fisher exact trigger vs diagnostic correctness `p=0.0036791367`; Mann–Whitney token-count difference `p=1.968701622239151e-24`.

**Interpretation:** natural latent trigger correlates with difficult/uncertain trajectories, but the comparison is observational and not causal.

**Conclusion:** KEEP.

---

## EXP-0004 — Paired latent-start suppression on naturally-triggered VStarBench examples
**Status:** COMPLETED FOR INTERVENTION; CORRECTNESS UTILITY INTERPRETATION INCONCLUSIVE PENDING RE-JUDGING

**Date:** 2026-09-13

**Branch:** `main`

**Scripts:** `scripts/10_vstar_latent_off_ablation.py`, `scripts/10_vstar_latent_off_ablation.sh`

**Script correction commit:** `3f7d42d0a556b22369de4992a8ecc13de6fa1c4f`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Test the causal effect of allowing Monet to enter latent mode on the exact examples that naturally triggered it at baseline.

**Intervention:** allow every model-vocabulary token ID except exactly `151666=<abs_vis_token>` using vLLM `allowed_token_ids`. Vocabulary size `151670`; allowed IDs `151669`, including latent-end ID `151667`.

**Failed implementation retained:** initial `bad_words` implementation failed before generation due vLLM 0.10.0 accessing missing `Qwen2TokenizerFast.max_token_id`. No scientific outputs were produced; environment unchanged.

**Intervention validation:**
```text
exact_exclusion_verified_samples: 72/72
block_verified_samples: 72/72
VSTAR_LATENT_OFF_ABLATION_PASS=True
```

**Raw diagnostic paired result:**
```text
baseline diagnostic correct: 34/72 = 0.472222
latent-off diagnostic correct: 24/72 = 0.333333
raw delta: -0.138889
correct->correct: 18
correct->wrong: 16
wrong->correct: 6
wrong->wrong: 32
McNemar exact two-sided p = 0.052478790283203125
mean tokens: 79.7778 -> 80.3333
```

**Category audit:**

`direct_attributes` (`n=44`):
```text
baseline diagnostic accuracy: 25/44 = 0.568182
latent-off diagnostic accuracy: 16/44 = 0.363636
delta: -0.204545
transitions: CC=12, CW=13, WC=4, WW=15
McNemar exact p = 0.049041748046875
mean tokens: 78.500 -> 76.909
```

`relative_position` (`n=28`):
```text
baseline diagnostic accuracy: 9/28 = 0.321429
latent-off diagnostic accuracy: 8/28 = 0.285714
delta: -0.035714
transitions: CC=6, CW=3, WC=2, WW=17
McNemar exact p = 1.0
mean tokens: 81.786 -> 85.714
```

**Critical post-hoc parser audit:**
- `correct->wrong=16`, but 13/16 latent-off predictions are `None` under the diagnostic option parser.
- `wrong->correct=6`, but 5/6 baseline predictions are `None` under the diagnostic option parser.
- Thus `18/22 = 81.8%` of discordant pairs involve parser failure on one side.
- Only four discordant pairs are explicit option-to-option changes: three `correct->wrong` (`14: B->A`, `29: B->A`, `112: D->A`) and one `wrong->correct` (`100: A->C`).
- Restricting to those four resolved discordant pairs gives exact McNemar `p=0.625`.

**Interpretation:** The decoding intervention is technically valid and causally removes access to latent-start. However, the apparent correctness benefit of latent access is heavily confounded by answer-format/parser failures. The raw 13.9-point drop and the direct-attributes `p=0.049` must not be treated as reliable evidence of utility until the stored raw outputs are re-judged with a more robust resolver or the intended API judge.

**Conclusion:** KEEP the intervention; utility conclusion INCONCLUSIVE pending robust re-judging.

**Next action:** Re-judge the stored raw outputs for all 22 discordant pairs, especially the 18 involving `None`, using conservative option-text matching / final-answer parsing. Recompute paired transitions afterward. No model rerun is needed.

---

## Experiment Template

```markdown
## EXP-XXXX — Short title
**Status:** RUNNING / COMPLETED / FAILED / ABORTED
**Date:** YYYY-MM-DD
**Branch:** `...`
**Commit:** `...`
**Tag:** `...`
**Base checkpoint:** `...`
**Config:** `...`
**Purpose:** ...
**Change from baseline:** ...
**Data / benchmark:** ...
**Settings:** ...
**Hardware:** ...
**Results:** ...
**Conclusion:** KEEP / REJECT / INCONCLUSIVE
**Reason:** ...
**Next action:** ...
```

## Reproducibility Rule
A result is not considered verified unless the exact commit, configuration, checkpoint, benchmark protocol, and numerical result are recorded here. Verified milestones should also receive a Git tag.