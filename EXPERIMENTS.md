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

**Changes:**
- Added `PROJECT_GOAL.md`.
- Added `CURRENT_STATE.md`.
- Added `DECISIONS.md`.
- Added `EXPERIMENTS.md`.

**Model run:** None.

**Metrics:** None.

**Conclusion:** Repository control structure initialized. No scientific result has been produced yet.

**Next action:** Reproduce and freeze the official Monet baseline before starting V0.

---

## EXP-0001 — VStarBench single-sample natural latent-trigger probe
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Script source commits:** `f541f02a2808b2a4230963c359276c1a01dcad39` (Python probe) and `c433e177e35f38e38bcbb22072aafe56a50483aa` (launcher)

**Base checkpoint:** local `models/Monet-7B` from `NOVAglow646/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Verify that the official Monet latent runner naturally activates on an unforced VStarBench example when driven through the pinned VLMEvalKit Qwen2VL/vLLM path, while observationally capturing raw vLLM token IDs.

**Change from baseline:** None to model generation semantics. The probe wraps `self.llm.generate` only to retain the returned vLLM object and inspect `o.outputs[0].token_ids`. No forced tokens, no `allowed_token_ids`, and no sampling modification were introduced.

**Data / benchmark:** `VStarBench`, sample position 0 / benchmark index 0.

**Prompt / target:**
- Question: `What is the material of the glove?`
- Options: A rubber / B cotton / C kevlar / D leather
- Ground truth: `A`
- Monet official system prompt was present in the final chat-template prompt.

**Inference settings:**
- `LATENT_SIZE=10`
- `temperature=0.0`
- `max_tokens=2048`
- `allowed_token_ids=None`
- vLLM 0.10.0 with the pinned Monet `monet_gpu_model_runner.py`
- tensor parallelism over four visible RTX 3090 GPUs selected by the launcher
- random seed: not applicable to greedy decoding

**Observed output:**
- final answer: `\\boxed{A. rubber}`
- ground-truth option: `A`
- generated tokens: 99
- latent start ID `151666` positions: `[25]`
- latent end ID `151667` positions: `[35]`
- detected latent segments: `[(25, 35)]`
- number of start tokens: 1
- number of end tokens: 1
- number of latent segments: 1
- `NATURAL_LATENT_TRIGGER=True`
- `VSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True`
- VLMEvalKit returned text matched raw candidate text exactly.

**Interpretation:** The first unforced VStarBench sample provides direct runtime evidence that Monet can naturally emit the latent-start token under the pinned evaluation path. The start-to-end position difference is 10, consistent with the configured `LATENT_SIZE=10` state-machine span. Text decoded from token IDs inside the latent span is not interpreted as semantic reasoning content, because Monet replaces the corresponding next-step input embeddings with cached last-layer representations during latent mode. This is a single-sample observation only and must not be reported as a benchmark-wide trigger rate.

**Conclusion:** KEEP. The observational raw-token capture path is valid enough to proceed to a multi-sample natural-trigger characterization without forced tokens.

**Next action:** Run a reproducible 20-sample VStarBench pilot, record per-sample raw token IDs/text and latent segment statistics, and compute `r_trigger = triggered_samples / total_samples`. Do not start V0 training yet.

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
