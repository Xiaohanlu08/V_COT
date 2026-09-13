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

`direct_attributes` (`n=44`): baseline diagnostic `25/44=0.568182`, latent-off `16/44=0.363636`, raw delta `-0.204545`, transitions `CC=12, CW=13, WC=4, WW=15`, McNemar `p=0.049041748046875`.

`relative_position` (`n=28`): baseline diagnostic `9/28=0.321429`, latent-off `8/28=0.285714`, raw delta `-0.035714`, transitions `CC=6, CW=3, WC=2, WW=17`, McNemar `p=1.0`.

**Critical parser confound:** 18/22 discordant pairs involve `None` on one side under the old diagnostic parser. The old parser is not suitable for final utility claims.

**Conclusion:** KEEP the intervention; utility conclusion INCONCLUSIVE pending robust re-judging.

---

## EXP-0005 — Manual audit of discordant paired outputs with parser `None`
**Status:** COMPLETED

**Date:** 2026-09-13

**Purpose:** Determine whether the 18 discordant baseline-vs-latent-off pairs containing a parser `None` represent genuine answer changes or answer-format/parser artifacts.

**Data:** stored raw outputs from `results/causal_ablation/vstar_latent_off_all_triggered.jsonl`; no model rerun.

**Audit result:**
- 16/18 apparent flips are parser/format artifacts. Baseline and latent-off give the same semantically correct answer, but one response uses answer text rather than an option letter, e.g. `\boxed{purple}`, `\boxed{silver}`, `\boxed{orange}`, `\boxed{left}`, `\boxed{right}`, or an unboxed semantic answer.
- Position 61 is a genuine `correct->wrong`: GT `A`; baseline says `white` (option A), while latent-off explicitly ends `FINAL ANSWER: C. golden`.
- Position 92 is a genuine `wrong->correct`: GT `D`; baseline says `brown` (option B), while latent-off gives black / option D.

**Additional parser failure discovered:** At position 61, the old parser returned latent-off prediction `A` even though the raw output explicitly says `FINAL ANSWER: C. golden`. The cause is the old regex scanning unrestricted prose for standalone letters A-D and matching an earlier article `a`. Therefore the old parser may be wrong even when it returns a non-`None` option.

**Provisional corrected paired table:** If only these 18 audited pairs are fixed while the remaining old labels are left untouched:
```text
correct -> correct: 34
correct -> wrong:   4
wrong   -> correct: 2
wrong   -> wrong:   32
McNemar exact two-sided p = 0.6875
```
This table is explicitly provisional and must not be used as the final utility result because non-`None` old-parser outputs can also be wrong.

**Conclusion:** The earlier apparent latent-utility signal is not reliable under the old parser. Full option-aware re-scoring of all 72 stored paired outputs is required before any correctness claim.

**Next action:** Run `scripts/11_rescore_paired_outputs.py` / `.sh`, which performs deterministic option-aware local re-scoring against the actual VStarBench option strings and leaves ambiguous cases unresolved for audit. No model rerun is required.

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