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

**Branch:** `main`

**Base checkpoint:** local `models/Monet-7B`

**Data:** 20 VStarBench samples selected without replacement with `VCOT_SEED=20260913`.

**Settings:** `LATENT_SIZE=10`, greedy decoding, no forced tokens.

**Result:**
- triggered: `8/20 = 0.40`
- balanced markers: 20/20
- multi-segment: 0/20
- all latent segments length 10
- diagnostic option accuracy: 14/20 = 0.70
- triggered mean output length 77.25 vs 37.00 non-triggered
- triggered diagnostic accuracy 3/8 vs 11/12 non-triggered

**Conclusion:** KEEP. Instrumentation was clean; full-dataset characterization justified.

---

## EXP-0003 — Full VStarBench natural latent-trigger characterization
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Scripts:** `scripts/09_vstar_natural_trigger_scan.py`, `scripts/09_vstar_natural_trigger_scan.sh`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Data:** complete VStarBench, all 191 examples.

**Settings:** `LATENT_SIZE=10`, greedy decoding, no forced tokens, official Monet system prompt.

**Primary results:**
- triggered: `72/191 = 0.3769633508`
- balanced markers: 191/191
- multi-segment: 0/191
- all 72 latent segments length 10
- overall diagnostic option accuracy: `116/191 = 0.60733`

**Triggered vs non-triggered:**
- triggered: `34/72 = 0.472222`, mean tokens `79.778`
- non-triggered: `82/119 = 0.689076`, mean tokens `45.773`
- Fisher exact two-sided `p=0.0036791367`
- Mann–Whitney two-sided `p=1.968701622239151e-24`

**Category-level:** trigger rates are similar for `direct_attributes` (38.3%) and `relative_position` (36.8%); within both categories, triggered examples remain longer and less often correct under the diagnostic parser.

**Interpretation:** natural latent trigger correlates with difficult/uncertain trajectories but observational comparison is not causal.

**Conclusion:** KEEP.

---

## EXP-0004 — Paired latent-start suppression on naturally-triggered VStarBench examples
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Scripts:** `scripts/10_vstar_latent_off_ablation.py`, `scripts/10_vstar_latent_off_ablation.sh`

**Script correction commit:** `3f7d42d0a556b22369de4992a8ecc13de6fa1c4f`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Test whether access to Monet's latent pathway has causal utility on the exact VStarBench examples that naturally entered latent mode at baseline.

**Baseline source:** `results/natural_trigger/vstar_n191_seed20260913.jsonl`; 72 examples had `triggered=True`.

**Intervention:** allow every model-vocabulary token ID except exactly `151666=<abs_vis_token>` using vLLM `allowed_token_ids`. Model vocabulary size is 151670, leaving 151669 allowed IDs, including latent-end ID 151667. Generation remains greedy and otherwise uses the same VStarBench prompt/model path.

**Failed implementation retained in history:** An initial implementation using `bad_words=["<abs_vis_token>"]` failed before generation because vLLM 0.10.0 `SamplingParams.update_from_tokenizer()` accesses `tokenizer.max_token_id`, absent on the installed `Qwen2TokenizerFast`. No scientific output was produced by that failed attempt. The environment was not changed.

**Smoke validation (`n=5`):**
- exact exclusion verified: 5/5
- output block verified: 5/5
- baseline diagnostic correct: 3/5
- latent-off diagnostic correct: 4/5
- `VSTAR_LATENT_OFF_ABLATION_PASS=True`

**Full paired run (`n=72`):**
```text
exact_exclusion_verified_samples: 72/72
block_verified_samples: 72/72
baseline diagnostic correct: 34/72 = 0.472222
latent-off diagnostic correct: 24/72 = 0.333333
accuracy delta (latent-off - baseline): -0.138889
```

**Paired transitions:**
```text
correct -> correct: 18
correct -> wrong:   16
wrong   -> correct: 6
wrong   -> wrong:   32
```

**Paired significance test:** exact two-sided McNemar `p=0.052478790283203125`.

**Output length:** baseline mean `79.7778` tokens; latent-off mean `80.3333` tokens.

**Interpretation:** Blocking the latent-start token on the same naturally-triggered examples causes a net 13.9 percentage-point decrease in diagnostic option accuracy. There are substantially more harmful flips under suppression (`correct->wrong=16`) than beneficial flips (`wrong->correct=6`). The McNemar p-value is narrowly above 0.05, so this is a strong directional trend rather than conventionally significant proof at the 5% threshold.

The paired result reverses the naive interpretation of EXP-0003: naturally-triggered examples are harder overall, but on those same examples access to latent mode appears beneficial on average. This supports viewing trigger status as a difficulty/uncertainty selector rather than evidence that latent reasoning is itself harmful.

**Scientific limitation:** The intervention establishes utility of allowing the latent pathway under this decoding policy. It does not establish that every latent hidden state is visually grounded, nor that natural latent emission is sufficient to define a good supervision target. The central V_COT hypothesis still requires direct positive/evidence-preserving versus negative/evidence-destroying visual-view tests on the latent tensors.

**Conclusion:** KEEP.

**Next action:** Analyze the 72 paired records by category/transition without rerunning the model; document the under-specified supplementary API judge used for formal Monet benchmark reproduction; then instrument exact latent tensors and run the no-training visual-evidence separability test before V0.

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
