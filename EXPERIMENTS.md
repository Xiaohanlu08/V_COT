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

**Result:** triggered `8/20=0.40`; all markers balanced; no multi-segment behavior; all latent segments length 10. Triggered examples were longer and less often correct under the original diagnostic parser than non-triggered examples.

**Conclusion:** KEEP for trigger characterization only. Later parser auditing showed the original correctness parser is unreliable for semantic answer text.

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

The original simple parser reported `116/191 = 0.60733` diagnostic accuracy, but this number is not suitable for scientific conclusions after the later parser audit.

**Conclusion:** KEEP for natural-trigger mechanics and frequency.

---

## EXP-0004 — Paired latent-start suppression on naturally-triggered VStarBench examples
**Status:** COMPLETED; INTERVENTION VALID, NO DETECTABLE CORRECTNESS EFFECT AFTER ROBUST LOCAL RESCORING

**Date:** 2026-09-13

**Branch:** `main`

**Scripts:** `scripts/10_vstar_latent_off_ablation.py`, `scripts/10_vstar_latent_off_ablation.sh`, `scripts/11_rescore_paired_outputs.py`, `scripts/11_rescore_paired_outputs.sh`

**Latent-off script correction commit:** `3f7d42d0a556b22369de4992a8ecc13de6fa1c4f`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Test the causal effect of allowing Monet to enter latent mode on the exact 72 VStarBench examples that naturally triggered it at baseline.

**Intervention:** allow every model-vocabulary token ID except exactly `151666=<abs_vis_token>` using vLLM `allowed_token_ids`. Vocabulary size `151670`; allowed IDs `151669`, including latent-end ID `151667`.

**Failed implementation retained:** initial `bad_words` implementation failed before generation due vLLM 0.10.0 accessing missing `Qwen2TokenizerFast.max_token_id`. No scientific outputs were produced; environment unchanged.

**Intervention validation:**
```text
exact_exclusion_verified_samples: 72/72
block_verified_samples: 72/72
VSTAR_LATENT_OFF_ABLATION_PASS=True
```

**Raw original-parser result (superseded for correctness interpretation):**
```text
baseline diagnostic correct: 34/72 = 0.472222
latent-off diagnostic correct: 24/72 = 0.333333
correct->correct: 18
correct->wrong: 16
wrong->correct: 6
wrong->wrong: 32
McNemar exact two-sided p = 0.052478790283203125
```

**Why the raw result was invalid for utility interpretation:**
Manual audit showed that 18/22 discordant pairs involved `None` on one side because the old parser only handled option-letter-style final answers reliably. Monet often emitted semantically equivalent answers such as `\boxed{purple}`, `\boxed{silver}`, or `\boxed{left}`. A non-`None` parsing error was also found at position 61: the old parser returned `A` although the raw latent-off output explicitly ended with `FINAL ANSWER: C. golden`.

**Option-aware rescoring:**
All 72 baseline/latent-off raw outputs were rescored deterministically using the actual VStarBench option strings. This scorer is local and deterministic; it is not Monet's under-specified supplementary API judge.

**Rescored result:**
```text
n: 72
changed_parse_samples: 49
baseline correct:   54/72 = 0.750000
latent-off correct: 54/72 = 0.750000
delta (latent-off - baseline): 0.000000
correct -> correct: 50
correct -> wrong:    4
wrong   -> correct:  4
wrong   -> wrong:   14
McNemar exact two-sided p = 1.0
unresolved baseline outputs: 0
unresolved latent-off outputs: 2 (positions 0, 34)
VSTAR_OPTION_AWARE_RESCORE_PASS=True
```

**Interpretation:** After correcting answer-format parsing, there is no detectable paired correctness effect of blocking latent entry on these 72 examples. The earlier apparent 13.9-point benefit of latent access was a parser artifact.

**Conclusion:** KEEP. Token-level utility line is closed for now; answer-level paired effect is neutral under the robust local scorer.

---

## EXP-0005 — Exact recurrent latent tensor capture on VStarBench position 0
**Status:** COMPLETED

**Date:** 2026-09-13

**Branch:** `main`

**Scripts:** `scripts/12_make_tensor_capture_runner.py`, `scripts/12_vstar_single_latent_tensor_probe.py`, `scripts/12_vstar_single_latent_tensor_probe.sh`

**Base checkpoint:** local `models/Monet-7B`

**Upstream Monet commit:** `08939998d3d643a73a316e349faa34f420429153`

**VLMEvalKit snapshot:** `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

**Purpose:** Capture the exact continuous recurrent hidden-state sequence used by Monet during a naturally triggered latent segment, while proving that the instrumentation does not alter generation.

**Instrumentation point:** A temporary copy of the pinned official Monet vLLM runner is patched immediately after:
```python
st["pending"] = last_token_h[i].detach()
```
The saved vector is therefore the same recurrent hidden state that is subsequently written into `self.inputs_embeds` for the next latent decode step. The live GPU tensor remains unchanged; only rank-0 float32 CPU copies are saved.

**Data:** VStarBench position 0, previously verified to naturally trigger one `LATENT_SIZE=10` segment.

**Validation:** The complete generated token ID sequence under instrumentation was compared with the previously saved natural baseline token sequence.

**Results:**
```text
generation_exact_match_baseline: true
num_latent_tensors: 10
hidden_size: 3584
dtype_saved: torch.float32
all_finite: true
mean_l2_norm: 288.1265869140625
min_l2_norm: 275.2210998535156
max_l2_norm: 293.09039306640625
adjacent_cosine_mean: 0.9219153655899895
adjacent_cosine_min: 0.6129838228225708
adjacent_cosine_max: 0.9985483288764954
VSTAR_SINGLE_LATENT_TENSOR_PROBE_PASS=True
```

**Saved outputs:**
```text
results/latent_capture/vstar_pos0_latents.pt
results/latent_capture/vstar_pos0_latents_summary.json
logs/12_vstar_single_latent_tensor_probe.log
```

**Interpretation:** Exact recurrent latent-state capture is now mechanically verified. The `10 x 3584` tensor sequence is the hidden-state trajectory actually fed recurrently by Monet. High mean adjacent cosine indicates a generally smooth trajectory, while the minimum adjacent cosine around `0.613` indicates at least one comparatively large state transition. These descriptive metrics do not by themselves establish visual grounding.

**Conclusion:** KEEP. This closes the tensor-access engineering gate.

**Next action:** Use V*Bench's benchmark-provided target-object/bounding-box annotations to construct a matched evidence-preserving and evidence-destroying pilot for position 0. Then implement fixed-prefix/fixed-trigger replay so `I`, `I+`, and `I-` share the same textual context and latent entry position. First require replay of the original image to reproduce the EXP-0005 latent trajectory before comparing visual interventions.

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