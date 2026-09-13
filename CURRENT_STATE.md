# Current State

## Project Stage
Baseline reproduction / natural latent-trigger characterization.

## Current Objective
Establish a reproducible Monet baseline before implementing any new latent-supervision method, then test whether naturally emitted latent states are useful rather than merely correlated with difficult examples.

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
- Protected Hugging Face stack remains unchanged after controlled dependency bring-up.

## Verified Monet Inference Path
- Official Monet example passes.
- latent start token `<abs_vis_token>` = `151666`.
- latent end token `</abs_vis_token>` = `151667`.
- official-style runner patch verified in parent and spawned workers.
- deterministic forced engineering diagnostic passed.
- natural VStarBench raw-token capture passed without forced tokens.

## VStarBench Evaluation Path
Pinned VLMEvalKit uses `Qwen2VLChat`. The relevant VStarBench path is the standard `ImageMCQDataset` prompt plus Monet's official system prompt. The vLLM generation path is greedy (`temperature=0.0`) with default `max_new_tokens=2048`. Normal VLMEvalKit discards `o.outputs[0].token_ids`; V_COT observationally captures those raw IDs without changing generation semantics.

## VStarBench Dataset
- total samples: 191
- categories: `direct_attributes` and `relative_position`
- dataset/prompt plumbing verified.

## Natural Latent Trigger Characterization — COMPLETE
### Single-sample probe
Unforced sample/index 0 naturally emitted one latent segment with start/end positions `[25]` and `[35]` under `LATENT_SIZE=10`. Recorded as `EXP-0001`.

### 20-sample pilot
- triggered: `8/20 = 0.40`
- all marker pairs balanced
- no multi-segment cases
- every latent segment length = 10
- exploratory pilot already suggested triggered samples were longer and less often correct under the diagnostic boxed-option parser.
Recorded as `EXP-0002`.

### Full 191-sample scan
Recorded as `EXP-0003`.

Core results:
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
VSTAR_NATURAL_TRIGGER_SCAN_PASS=True
```

Because all 191 examples in this pinned VStarBench version were evaluated, 37.7% is the exact benchmark proportion for this run, not a sample estimate of this finite benchmark.

## Full JSONL Trigger-vs-Nontrigger Analysis — VERIFIED
### Overall
Triggered (`n=72`):
```text
heuristic_correct: 34/72 = 0.472222
mean_generated_tokens: 79.778
median_generated_tokens: 76
range: 50–116
```

Non-triggered (`n=119`):
```text
heuristic_correct: 82/119 = 0.689076
mean_generated_tokens: 45.773
median_generated_tokens: 45
range: 17–94
```

Exploratory association tests:
```text
trigger x diagnostic correctness:
Fisher odds ratio = 0.4037227214
Fisher two-sided p = 0.0036791367

generated-token count:
Mann–Whitney U = 8060.5
Mann–Whitney two-sided p = 1.968701622239151e-24
```

### Category level
`direct_attributes` (`n=115`):
```text
trigger rate: 44/115 = 0.382609
overall diagnostic accuracy: 81/115 = 0.704348
triggered diagnostic accuracy: 25/44 = 0.568182
non-triggered diagnostic accuracy: 56/71 = 0.788732
triggered mean tokens: 78.500
non-triggered mean tokens: 42.014
```

`relative_position` (`n=76`):
```text
trigger rate: 28/76 = 0.368421
overall diagnostic accuracy: 35/76 = 0.460526
triggered diagnostic accuracy: 9/28 = 0.321429
non-triggered diagnostic accuracy: 26/48 = 0.541667
triggered mean tokens: 81.786
non-triggered mean tokens: 51.333
```

The trigger rates are similar across the two categories (38.3% vs 36.8%), while the triggered-vs-nontriggered correctness and output-length differences occur in both categories. Therefore the overall association is not plausibly explained only by the category mixture.

## Scientific Interpretation
The full-dataset observation strongly supports the following restricted claim: under the pinned Monet/VStarBench evaluation path, natural latent triggering is associated with longer reasoning trajectories and lower diagnostic option correctness.

It does **not** establish that latent reasoning causes errors. Triggering is endogenous: Monet decides when to emit the latent-start token, so it may be a marker of task difficulty, uncertainty, or a difficult reasoning trajectory. The current evidence therefore argues against treating every naturally emitted latent state as automatically useful supervision.

This is directly relevant to V_COT's intended contribution: latent states should be selected or weighted according to visual evidence and utility, rather than supervised simply because Monet emitted them.

## Baseline Reproduction Status
Natural-trigger characterization is complete. Formal benchmark reproduction is not yet complete because Monet's README specifies a supplementary API judge for reported benchmark scores. The current boxed-option extractor is diagnostic only and must not be compared directly with Monet's reported VStarBench score.

## Next Milestones
- [x] Pin Monet implementation and checkpoint.
- [x] Reproduce runtime and official example.
- [x] Verify latent token IDs and official runner patch.
- [x] Verify forced latent runtime path as an engineering diagnostic.
- [x] Bring up pinned VLMEvalKit without perturbing Monet core versions.
- [x] Verify VStarBench dataset/prompt path.
- [x] Implement observational raw-token capture.
- [x] Verify unforced natural latent trigger.
- [x] Run 20-sample trigger pilot.
- [x] Run all 191 VStarBench examples.
- [x] Analyze trigger status vs category, diagnostic correctness, and response length.
- [ ] Reproduce the official VStarBench baseline score under Monet's documented supplementary-judge protocol.
- [ ] Freeze the reproduced baseline with a Git tag.
- [ ] Design and run a causal latent-suppression ablation before interpreting latent utility.
- [ ] Locate/instrument exact latent-state tensors for the no-training positive/negative visual-evidence separability test.
- [ ] Start V0 only after those gates pass.

## Proposed Next Scientific Test
Before V0 training, perform a paired causal ablation using the same VStarBench prompts and greedy decoding:
1. baseline Monet generation;
2. latent-suppressed generation where only `<abs_vis_token>` is forbidden at decoding time while all other tokens remain available.

vLLM 0.10.0 supports `bad_words`, which can forbid a specified token sequence. For a clean implementation, first verify locally that `bad_words=["<abs_vis_token>"]` maps exactly to token ID `151666` and does not block `</abs_vis_token>` or alter unrelated tokens. This intervention should be treated as a causal decoding ablation, not as the reproduced baseline.

Primary paired outcomes should be answer changes and correctness changes on the 72 examples that naturally triggered at baseline; all 191 examples can be retained as a secondary analysis. A forced-latent condition on naturally non-triggered examples is more invasive and should not be the first causal test.

## Known Issues
- GPU server cannot directly access GitHub; use local staging/direct handoff.
- VLMEvalKit was transferred as an archive and therefore has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.

## Next Action
Do not start V0 training yet. First reproduce the official VStarBench score with Monet's documented supplementary-judge protocol, then run the latent-suppression causal ablation. After the reproduced baseline is frozen, proceed to exact latent-tensor instrumentation and the positive/negative visual-evidence separability test.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
