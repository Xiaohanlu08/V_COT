# Current State

## Project Stage
Baseline reproduction / latent-utility characterization.

## Current Objective
Establish a reproducible Monet baseline and determine whether naturally emitted latent states are useful, visually grounded, and suitable targets for selective supervision.

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
- Pinned VLMEvalKit imports as `0.2rc1`
- Protected Hugging Face stack remains unchanged.

## Verified Monet Inference Path
- Official Monet example passes.
- `<abs_vis_token>` = `151666`; `</abs_vis_token>` = `151667`.
- Official-style runner patch verified in parent and spawned workers.
- Deterministic forced engineering diagnostic passed.
- Natural VStarBench raw-token capture passed without forced tokens.

## VStarBench Evaluation Path
Pinned VLMEvalKit uses `Qwen2VLChat`. VStarBench follows the standard `ImageMCQDataset` prompt plus Monet's official system prompt. The vLLM path is greedy (`temperature=0.0`) with `max_new_tokens=2048`. V_COT captures raw token IDs observationally without altering the baseline generation semantics.

## Natural Latent Trigger Characterization — COMPLETE
Recorded as `EXP-0001` through `EXP-0003`.

Full 191-sample result:
```text
samples: 191
triggered_samples: 72
trigger_rate: 72/191 = 0.3769633508 (~37.7%)
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

Exploratory associations:
```text
Fisher exact trigger vs diagnostic correctness: p = 0.0036791367
Mann–Whitney generated-token count: p = 1.968701622239151e-24
```

Category trigger rates are similar (`direct_attributes` 38.3%, `relative_position` 36.8%), and the same triggered-vs-nontriggered directional accuracy gap appears in both categories. This is observational only; trigger status is endogenous and may mark difficult or uncertain examples.

## Paired Latent-Suppression Ablation — COMPLETE
Recorded as `EXP-0004`.

### Intervention implementation
The first attempted `bad_words=["<abs_vis_token>"]` implementation failed before generation because vLLM 0.10.0 `SamplingParams.update_from_tokenizer()` accesses `tokenizer.max_token_id`, which is absent from the installed `Qwen2TokenizerFast`. No model outputs were produced by that failed attempt.

The validated implementation uses vLLM `allowed_token_ids` to allow the entire model vocabulary except exactly token ID `151666` (`<abs_vis_token>`). Model vocabulary size is `151670`, so `151669` IDs remain allowed, including latent-end ID `151667`.

A 5-sample smoke test passed with:
```text
exact_exclusion_verified_samples: 5/5
block_verified_samples: 5/5
VSTAR_LATENT_OFF_ABLATION_PASS=True
```

### Full paired intervention on all 72 naturally-triggered baseline samples
```text
n_intervened: 72
exact_exclusion_verified_samples: 72
block_verified_samples: 72
baseline diagnostic correct: 34/72 = 0.472222
latent-off diagnostic correct: 24/72 = 0.333333
accuracy delta (latent-off - baseline): -0.138889
```

Paired transitions:
```text
correct -> correct: 18
correct -> wrong:   16
wrong   -> correct: 6
wrong   -> wrong:   32
```

Exact two-sided McNemar test:
```text
p = 0.052478790283203125
```

Output length:
```text
baseline mean tokens: 79.7778
latent-off mean tokens: 80.3333
```

### Interpretation
Blocking the latent-start token reduces the diagnostic option accuracy by 13.9 percentage points on the exact set of samples that naturally triggered latent mode at baseline. Suppression causes more `correct->wrong` flips (16) than `wrong->correct` flips (6). The paired McNemar result is suggestive but narrowly above the conventional 0.05 threshold, so this should be described as evidence of a beneficial trend, not a conventionally significant proof.

This paired intervention is substantially stronger than the earlier triggered-vs-nontriggered observational comparison: for the same naturally-triggered examples, removing access to latent mode often harms the answer. Therefore the earlier lower accuracy of naturally-triggered examples is better interpreted as difficulty/uncertainty selection rather than evidence that latent reasoning is generally harmful.

However, this intervention only establishes utility of access to the latent pathway under this decoding policy. It does **not** establish that every latent hidden state is visually grounded, nor that all emitted latent states are useful supervision targets. The next scientific question remains whether positive/evidence-preserving views produce latent states that are more aligned/useful than evidence-destroying views.

## Formal Baseline Reproduction Caveat
Monet's README states that exact-matching evaluation was replaced by an API judge and asks users to apply an API model as a supplementary judge, but the README does not specify the judge model in the evaluation section. Therefore exact reproduction of the reported VStarBench score is currently under-specified unless the judge model/configuration can be recovered from code, paper, or authors. The current boxed-option extraction remains diagnostic only and must not be compared directly with the paper's reported score.

## Next Milestones
- [x] Pin Monet implementation and checkpoint.
- [x] Reproduce runtime and official inference path.
- [x] Verify latent token IDs and runner patch.
- [x] Bring up pinned VLMEvalKit.
- [x] Verify VStarBench dataset/prompt path.
- [x] Capture natural raw latent markers.
- [x] Characterize all 191 VStarBench examples.
- [x] Analyze trigger status vs correctness/category/response length.
- [x] Implement and validate exact latent-start suppression.
- [x] Run paired latent-off ablation on all 72 naturally-triggered examples.
- [ ] Recover or document the supplementary-judge protocol sufficiently for formal baseline-score reproduction.
- [ ] Freeze the reproduced baseline with a Git tag once scoring is documented.
- [ ] Locate/instrument the exact latent-state tensors.
- [ ] Run no-training positive/evidence-preserving vs negative/evidence-destroying visual-view latent separability test.
- [ ] Start V0 only if the separability/utility gate is supported.

## Known Issues
- GPU server cannot directly access GitHub; use local staging/direct handoff.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.
- Monet evaluation README requires an API judge but does not name the exact judge model in the evaluation section.

## Next Action
First analyze the paired latent-off JSONL by benchmark category and by transition type without rerunning the model. In parallel, inspect Monet/VLMEvalKit/paper sources for the exact supplementary-judge model and scoring configuration. After the paired analysis and baseline-scoring protocol are documented, move to exact latent-tensor instrumentation and the no-training positive/negative visual-evidence separability test. Do not start V0 training yet.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
