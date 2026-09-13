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

## Natural Latent Trigger Characterization — COMPLETE
Recorded as `EXP-0001` through `EXP-0003`.

Full VStarBench (`n=191`):
```text
triggered_samples: 72/191 = 0.3769633508 (~37.7%)
balanced_marker_samples: 191/191
multi_segment_samples: 0/191
all 72 latent segments have length 10
overall diagnostic option accuracy: 116/191 = 0.60733
```

Triggered (`n=72`): diagnostic correct `34/72 = 0.472222`, mean tokens `79.778`.
Non-triggered (`n=119`): diagnostic correct `82/119 = 0.689076`, mean tokens `45.773`.

Observational association tests:
```text
Fisher exact trigger vs diagnostic correctness: p = 0.0036791367
Mann–Whitney generated-token count: p = 1.968701622239151e-24
```

Category trigger rates are similar (`direct_attributes` 38.3%, `relative_position` 36.8%). Trigger status is endogenous, so these observational differences do not establish latent utility or harm.

## Paired Latent-Suppression Ablation — INTERVENTION VERIFIED, UTILITY JUDGMENT NOT YET FINAL
Recorded as `EXP-0004`.

### Intervention implementation
The validated implementation uses vLLM `allowed_token_ids` to allow the full model vocabulary except exactly token ID `151666=<abs_vis_token>`. Model vocabulary size is `151670`, leaving `151669` allowed IDs including latent-end ID `151667`.

The original `bad_words` implementation failed before generation because vLLM 0.10.0 calls `tokenizer.max_token_id`, which is absent from the installed `Qwen2TokenizerFast`. The environment was not modified.

Full intervention on all 72 naturally-triggered baseline samples:
```text
exact_exclusion_verified_samples: 72/72
block_verified_samples: 72/72
baseline diagnostic correct: 34/72 = 0.472222
latent-off diagnostic correct: 24/72 = 0.333333
raw diagnostic delta: -0.138889
correct -> correct: 18
correct -> wrong:   16
wrong   -> correct: 6
wrong   -> wrong:   32
McNemar exact two-sided p = 0.052478790283203125
baseline mean tokens: 79.7778
latent-off mean tokens: 80.3333
```

### Category-level paired analysis
`direct_attributes` (`n=44`):
```text
baseline diagnostic accuracy: 25/44 = 0.568182
latent-off diagnostic accuracy: 16/44 = 0.363636
delta: -0.204545
correct->wrong: 13
wrong->correct: 4
McNemar exact p = 0.049041748046875
mean tokens: 78.500 -> 76.909
```

`relative_position` (`n=28`):
```text
baseline diagnostic accuracy: 9/28 = 0.321429
latent-off diagnostic accuracy: 8/28 = 0.285714
delta: -0.035714
correct->wrong: 3
wrong->correct: 2
McNemar exact p = 1.0
mean tokens: 81.786 -> 85.714
```

### Critical parser confound discovered
The category audit exposed that most discordant correctness flips are not clean option-to-option changes under the current diagnostic parser:
- of 16 `correct->wrong` flips, 13 have `latent_off_predicted_option_diagnostic=None`;
- of 6 `wrong->correct` flips, 5 have `baseline_predicted_option_diagnostic=None`;
- therefore 18/22 (81.8%) discordant pairs involve a parser failure (`None`) on one side.

Only four discordant pairs are explicit option-to-option changes under the current parser:
- `correct->wrong`: 3 (`14: B->A`, `29: B->A`, `112: D->A`);
- `wrong->correct`: 1 (`100: A->C`).
For these four resolved discordant pairs alone, exact McNemar is not significant (`p=0.625`).

Therefore the previously observed 13.9-point raw diagnostic drop and the direct-attributes `p=0.049` **cannot yet be treated as reliable evidence that latent access improves answer correctness**. They may partly or largely reflect answer-format / parser failures. The intervention itself is valid, but the utility conclusion is pending robust re-judging of the stored raw outputs.

## Formal Baseline Reproduction Caveat
Monet's README says exact matching was replaced by an API judge and instructs users to apply an API model as a supplementary judge, but the evaluation section does not identify the exact judge model/configuration. Current boxed-option extraction is diagnostic only and must not be compared directly with the reported Monet VStarBench score.

## Next Milestones
- [x] Pin Monet implementation and checkpoint.
- [x] Reproduce runtime and official inference path.
- [x] Verify latent token IDs and runner patch.
- [x] Bring up pinned VLMEvalKit.
- [x] Verify VStarBench dataset/prompt path.
- [x] Characterize natural latent triggering over all 191 examples.
- [x] Implement and validate exact latent-start suppression.
- [x] Run paired latent-off intervention over all 72 naturally-triggered examples.
- [x] Analyze paired intervention by category and transition type.
- [x] Identify diagnostic-parser confounding in the paired correctness analysis.
- [ ] Re-judge stored baseline/latent-off raw outputs with a more robust deterministic MCQ resolver and audit all previously `None` cases.
- [ ] Recover or document the supplementary API judge protocol sufficiently for formal baseline-score reproduction.
- [ ] Freeze the reproduced baseline with a Git tag once scoring is documented.
- [ ] Locate/instrument exact latent-state tensors.
- [ ] Run no-training positive/evidence-preserving vs negative/evidence-destroying visual-view latent separability test.
- [ ] Start V0 only if the separability/utility gate is supported.

## Known Issues
- GPU server cannot directly access GitHub; use local staging/direct handoff.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.
- Monet evaluation README requires an API judge but does not identify the exact judge model in the evaluation section.

## Next Action
Do not rerun the model yet. First inspect and re-judge the stored raw texts for the 22 discordant paired cases, especially the 18 cases involving `None`, using the VStarBench option strings and conservative final-answer parsing. Recompute the paired transition table only after this audit. Then move to exact latent-tensor instrumentation and the positive/negative visual-evidence separability test. Do not start V0 training yet.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.