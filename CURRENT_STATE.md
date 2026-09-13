# Current State

## Project Stage
Baseline reproduction / natural latent-trigger characterization / causal latent-utility probing.

## Current Objective
Establish a reproducible Monet baseline and determine whether naturally emitted latent states are useful rather than merely correlated with difficult examples, before implementing V0 latent supervision.

## Current Branch
`main`

## Selected Upstream Baseline
- Monet: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Checkpoint: `NOVAglow646/Monet-7B`
- VLMEvalKit reproducibility snapshot chosen by V_COT: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`

## Verified Runtime
- Python 3.10.21
- `torch==2.7.1+cu126`
- `torchvision==0.22.1+cu126`
- `transformers==4.54.0`
- `vllm==0.10.0`
- `trl==0.15.2`
- Monet-7B checkpoint locally verified
- pinned VLMEvalKit imports as `0.2rc1`
- protected Hugging Face stack remains unchanged after controlled dependency bring-up.

## Verified Monet Inference Path
- official Monet example passes;
- `<abs_vis_token>` = `151666`;
- `</abs_vis_token>` = `151667`;
- official-style runner patch verified in parent and spawned workers;
- deterministic forced engineering diagnostic passed;
- natural VStarBench raw-token capture passed without forced tokens.

## VStarBench Evaluation Path
Pinned VLMEvalKit uses `Qwen2VLChat`. VStarBench follows the standard `ImageMCQDataset` prompt plus Monet's official system prompt. The vLLM generation path is greedy (`temperature=0.0`) with `max_new_tokens=2048`. V_COT observationally captures raw vLLM token IDs without otherwise changing baseline generation semantics.

## Natural Latent Trigger Characterization — COMPLETE
Full benchmark run over all 191 examples (`EXP-0003`):
```text
triggered_samples: 72/191 = 0.3769633508 (~37.7%)
balanced_marker_samples: 191/191
multi_segment_samples: 0/191
total_latent_segments: 72
all observed latent segment lengths: 10
mean_generated_tokens overall: 58.5916
heuristic_option_correct: 116/191 = 0.60733 (diagnostic only)
```

Triggered (`n=72`):
```text
heuristic_correct: 34/72 = 0.472222
mean_generated_tokens: 79.778
```

Non-triggered (`n=119`):
```text
heuristic_correct: 82/119 = 0.689076
mean_generated_tokens: 45.773
```

Exploratory association tests:
```text
Fisher exact OR = 0.4037227214
Fisher two-sided p = 0.0036791367
Mann–Whitney U = 8060.5
Mann–Whitney two-sided p = 1.968701622239151e-24
```

The same directional correctness gap appears within both benchmark categories, while category trigger rates are similar (`direct_attributes` 38.3%, `relative_position` 36.8%). This supports the restricted observational claim that natural latent triggering is associated with longer trajectories and lower diagnostic correctness; it does not show that latent reasoning causes errors.

## Step 10 Paired Latent-Off Intervention — SMOKE TEST VERIFIED
Purpose: causally intervene on examples that naturally triggered at baseline by forbidding only the latent-start token while preserving greedy decoding and all other model-vocabulary tokens.

### First implementation attempt — FAILED CLEANLY
The first implementation used vLLM 0.10.0 `bad_words=["<abs_vis_token>"]`. It failed before generation because `SamplingParams.update_from_tokenizer()` accesses `tokenizer.max_token_id`, which is absent on the installed `Qwen2TokenizerFast`. No model result was produced from this failed path. The runtime environment was not changed.

### Corrected implementation
`scripts/10_vstar_latent_off_ablation.py` was revised to use vLLM `allowed_token_ids` containing every model-vocabulary token ID except `151666`.

Verified intervention invariants:
- `model_vocab_size=151670`;
- allowed token count = `151669`;
- latent start ID `151666` is excluded;
- latent end ID `151667` remains allowed;
- generation output is rejected if `151666` appears;
- baseline source is the complete Step 09 JSONL;
- only baseline-triggered examples are selected.

### 5-sample smoke result
Selected baseline-triggered positions: `[0, 1, 2, 3, 6]`.

```text
n_intervened: 5
exact_exclusion_verified_samples: 5
block_verified_samples: 5
baseline diagnostic correct: 3/5 = 0.60
latent-off diagnostic correct: 4/5 = 0.80
accuracy delta: +0.20
transitions:
  correct->correct: 3
  correct->wrong: 0
  wrong->correct: 1
  wrong->wrong: 1
McNemar exact two-sided p: 1.0
mean tokens baseline: 82.4
mean tokens latent-off: 72.0
VSTAR_LATENT_OFF_ABLATION_PASS=True
```

Interpretation: the smoke test validates the intervention mechanics only. With five examples and one discordant correctness flip, the apparent +20 pp accuracy change is not evidence of benefit (`p=1.0`). The intervention is now approved for the full set of 72 baseline-triggered examples.

## Baseline Reproduction Status
Natural-trigger characterization is complete. Formal benchmark-score reproduction is not yet complete because Monet's README specifies a supplementary API judge; V_COT's boxed-option extraction remains diagnostic only.

## Next Milestones
- [x] Pin Monet implementation and checkpoint.
- [x] Reproduce runtime and official example.
- [x] Verify latent token IDs and official runner patch.
- [x] Bring up pinned VLMEvalKit without perturbing Monet core versions.
- [x] Verify VStarBench dataset/prompt path.
- [x] Characterize natural latent triggering on all 191 examples.
- [x] Analyze trigger status vs category, diagnostic correctness, and response length.
- [x] Implement and verify a clean 5-sample latent-off paired intervention.
- [ ] Run the latent-off intervention on all 72 baseline-triggered examples.
- [ ] Analyze paired correctness flips, answer changes, output length, and category-level effects.
- [ ] Reproduce the official VStarBench baseline score under Monet's documented supplementary-judge protocol.
- [ ] Freeze the reproduced baseline with a Git tag.
- [ ] Locate/instrument exact latent-state tensors for the no-training positive/negative visual-evidence separability test.
- [ ] Start V0 only after these gates pass.

## Known Issues
- GPU server cannot directly access GitHub; use local staging/direct handoff.
- VLMEvalKit was transferred as an archive and therefore has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.
- vLLM 0.10.0 `bad_words` preprocessing is incompatible with the installed `Qwen2TokenizerFast` because it expects `max_token_id`; V_COT does not use that path for Step 10.

## Next Action
Run the corrected Step 10 script with `VCOT_N=0` to intervene on all 72 examples that naturally triggered in the baseline. Do not alter the script or sampling settings. After the run, inspect `exact_exclusion_verified_samples`, `block_verified_samples`, the four paired correctness transitions, McNemar exact p-value, and baseline-vs-latent-off output lengths. Do not start V0 training yet.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
