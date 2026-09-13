# Experiment Log

This file is the permanent record of experiments that were actually run. Planned experiments should stay in `CURRENT_STATE.md` until execution begins.

## Recording Rules
For every experiment, record:
- experiment ID,
- date,
- branch,
- commit SHA,
- base checkpoint,
- configuration file,
- dataset / benchmark,
- training settings,
- random seed,
- hardware,
- result metrics,
- comparison baseline,
- conclusion,
- next action.

A failed experiment must be recorded as carefully as a successful one. Do not delete failed runs from project history.

---

## EXP-0000 — Repository Initialization
**Status:** COMPLETED

**Purpose:** Establish project governance and reproducibility anchors before importing or modifying model code.

**Changes:** Added `PROJECT_GOAL.md`, `CURRENT_STATE.md`, `DECISIONS.md`, and `EXPERIMENTS.md`.

**Conclusion:** Repository control structure initialized.

---

## EXP-0001 — VStarBench single-sample natural latent-trigger probe
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Script source commits:** `f541f02a2808b2a4230963c359276c1a01dcad39` and `c433e177e35f38e38bcbb22072aafe56a50483aa`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Result:** Unforced VStarBench sample 0 naturally emitted one latent segment with start/end positions `[25]` and `[35]` under `LATENT_SIZE=10`; final option was correct; `VSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True`.

**Conclusion:** KEEP. Natural latent activation is directly observable on the real evaluation path.

---

## EXP-0002 — VStarBench 20-sample natural latent-trigger pilot
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Script source commits:** `505f0b17590ba0105905f8dee9a671f74639733f` and `58dad2fc9c2d0c980b16ea7ff478e81fbde50f25`

**Result:**
- trigger rate `8/20 = 0.40`;
- marker balance `20/20`;
- no multi-segment cases;
- triggered diagnostic correct `3/8`, mean tokens `77.25`;
- non-triggered diagnostic correct `11/12`, mean tokens `37.00`.

**Conclusion:** KEEP. Instrumentation is clean and full-benchmark characterization is justified.

---

## EXP-0003 — Full VStarBench natural latent-trigger characterization
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Scripts:** `scripts/09_vstar_natural_trigger_scan.py`, `scripts/09_vstar_natural_trigger_scan.sh`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Data / benchmark:** Complete VStarBench, all 191 examples.

**Inference settings:** `LATENT_SIZE=10`, `temperature=0.0`, no forced tokens, official Monet system prompt.

**Primary results:**
- triggered `72/191 = 0.3769633508`;
- balanced markers `191/191`;
- multi-segment `0/191`;
- all 72 latent segments length 10;
- diagnostic heuristic option correct `116/191 = 0.60733`.

Triggered (`n=72`): diagnostic correct `34/72 = 0.472222`, mean tokens `79.778`.

Non-triggered (`n=119`): diagnostic correct `82/119 = 0.689076`, mean tokens `45.773`.

Exploratory association tests:
- Fisher exact OR `0.4037227214`, two-sided `p=0.0036791367`;
- Mann–Whitney U `8060.5`, two-sided `p=1.968701622239151e-24`.

**Interpretation:** Natural trigger status is associated with longer trajectories and lower diagnostic correctness but is endogenous, so the result is not causal evidence that latent reasoning harms answers.

**Conclusion:** KEEP. Natural latent emission itself should not be treated as a proxy for useful supervision.

---

## EXP-0004 — Paired latent-off intervention smoke test
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Corrected script commit:** `3f7d42d0a556b22369de4992a8ecc13de6fa1c4f`

**Scripts:** `scripts/10_vstar_latent_off_ablation.py`, `scripts/10_vstar_latent_off_ablation.sh`

**Base checkpoint:** local `models/Monet-7B`

**Baseline source:** `results/natural_trigger/vstar_n191_seed20260913.jsonl`

**Purpose:** Verify a clean paired causal intervention that suppresses natural latent entry by forbidding only `<abs_vis_token>` (`151666`) while retaining greedy decoding and all other model-vocabulary token IDs.

**Implementation history:**
- Initial `bad_words=["<abs_vis_token>"]` implementation FAILED CLEANLY before generation because vLLM 0.10.0 `SamplingParams.update_from_tokenizer()` accesses `tokenizer.max_token_id`, which is absent on the installed `Qwen2TokenizerFast`.
- No scientific result was produced from the failed path and no package/runtime change was made.
- Corrected implementation uses `allowed_token_ids` containing every model-vocabulary ID except `151666`.

**Smoke-test data:** First 5 baseline-triggered positions `[0, 1, 2, 3, 6]`.

**Verified intervention mechanics:**
- `model_vocab_size=151670`;
- `allowed_token_count=151669`;
- `exact_exclusion_verified_samples=5/5`;
- `block_verified_samples=5/5`;
- latent end token remains allowed;
- `VSTAR_LATENT_OFF_ABLATION_PASS=True`.

**Diagnostic paired results:**
- baseline correct `3/5 = 0.60`;
- latent-off correct `4/5 = 0.80`;
- delta `+0.20`;
- correct->correct: 3;
- correct->wrong: 0;
- wrong->correct: 1;
- wrong->wrong: 1;
- McNemar exact two-sided `p=1.0`;
- mean tokens baseline `82.4`;
- mean tokens latent-off `72.0`.

**Interpretation:** This smoke test validates intervention mechanics, not latent utility. Five samples with one discordant correctness flip are far too small to support a benefit/harm claim; `p=1.0` is explicitly inconclusive.

**Conclusion:** KEEP. The corrected latent-off intervention is mechanically verified and approved for all 72 baseline-triggered examples.

**Next action:** Run `VCOT_N=0 bash scripts/10_vstar_latent_off_ablation.sh`, then analyze all paired correctness transitions, answer changes, category effects, and output-length changes. Do not start V0 training yet.

---

## Experiment Template

```markdown
## EXP-XXXX — Short title
**Status:** RUNNING / COMPLETED / FAILED / ABORTED

**Date:** YYYY-MM-DD

**Branch:** `...`

**Commit:** `...`

**Tag:** `...` (if verified)

**Base checkpoint:** `...`

**Config:** `configs/...`

**Purpose:**
...

**Change from baseline:**
...

**Data / benchmark:**
...

**Training settings:**
- learning rate:
- epochs / steps:
- batch size:
- precision:
- seed:
- other:

**Hardware:**
...

**Results:**
| Metric | Baseline | This run | Delta |
|---|---:|---:|---:|
| ... | ... | ... | ... |

**Conclusion:** KEEP / REJECT / INCONCLUSIVE

**Reason:**
...

**Next action:**
...
```

## Reproducibility Rule
A result is not considered verified unless the exact commit, configuration, checkpoint, benchmark protocol, and numerical result are recorded here. Verified milestones should also receive a Git tag.
