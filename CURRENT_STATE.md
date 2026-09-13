# Current State

## Project Stage
Baseline reproduction / latent-utility characterization.

## Current Objective
Establish a reproducible Monet baseline and determine whether naturally emitted latent states are visually grounded and suitable targets for selective supervision.

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
```

The original simple boxed-option parser produced diagnostic accuracy `116/191 = 0.60733`, but later auditing showed that this parser is not reliable for Monet outputs that place semantic answer text rather than the option letter inside `\boxed{}`. Therefore old diagnostic accuracy values are retained only as historical diagnostics and should not be used for scientific conclusions.

## Paired Latent-Suppression Ablation — INTERVENTION VERIFIED; NO DETECTABLE CORRECTNESS EFFECT AFTER RESCORING
Recorded as `EXP-0004` and rescored with `scripts/11_rescore_paired_outputs.py`.

### Intervention implementation
The validated intervention uses vLLM `allowed_token_ids` to allow the full model vocabulary except exactly token ID `151666=<abs_vis_token>`. Model vocabulary size is `151670`, leaving `151669` allowed IDs including latent-end ID `151667`.

Intervention integrity over all 72 naturally-triggered examples:
```text
exact_exclusion_verified_samples: 72/72
block_verified_samples: 72/72
VSTAR_LATENT_OFF_ABLATION_PASS=True
```

### Parser audit
The original diagnostic parser was heavily confounded by answer formatting:
- 18/22 originally discordant pairs contained `None` on one side.
- Manual inspection showed that most such cases expressed the same semantic answer using option text rather than an option letter.
- `position 61` also exposed a non-`None` parser error: the old parser reported `A`, while the raw latent-off output explicitly ended with `FINAL ANSWER: C. golden`.

This motivated deterministic option-aware rescoring of all 72 paired baseline/latent-off raw outputs using the real VStarBench option strings.

### Option-aware rescoring result
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
unresolved latent-off outputs: 2 (positions 0 and 34)
VSTAR_OPTION_AWARE_RESCORE_PASS=True
```

The two unresolved latent-off cases cannot overturn the main paired conclusion: even if both were ultimately correct, the latent-off accuracy would increase only to `56/72 = 0.7778`, and the discordant-pair balance would still not provide evidence that suppressing latent entry systematically helps or harms correctness.

### Scientific interpretation
The latent-start suppression intervention is technically clean, but after correcting answer-format parsing there is **no detectable paired correctness effect** on these 72 naturally-triggered VStarBench examples (`54/72` vs `54/72`, McNemar `p=1.0`). Therefore the earlier apparent 13.9-point drop under latent suppression was a parser artifact and must not be used as evidence that latent access improves answer correctness.

This does not imply that Monet's latent states are useless. It means token-level answer accuracy under this intervention is approximately neutral at this scale. The central unanswered question is now more specific and more important for V_COT: whether the latent hidden states themselves encode and respond to task-relevant visual evidence.

## Formal Baseline Reproduction Caveat
Monet's README says exact matching was replaced by an API judge and instructs users to apply an API model as a supplementary judge, but the evaluation section does not identify the exact judge model/configuration. Current local option-aware rescoring is deterministic and substantially more reliable than the original parser, but it is still not the official supplementary API judge and should not be compared directly with the paper's reported VStarBench score.

## Step 12 — Exact Latent Tensor Capture Probe IMPLEMENTED, NOT YET VERIFIED
Implemented files:
- `scripts/12_make_tensor_capture_runner.py`
- `scripts/12_vstar_single_latent_tensor_probe.py`
- `scripts/12_vstar_single_latent_tensor_probe.sh`

The probe copies the pinned official Monet runner into a temporary directory and inserts observation-only dumping exactly where Monet assigns `st["pending"] = last_token_h[i].detach()`. These are the recurrent hidden-state vectors subsequently consumed through `self.inputs_embeds.index_copy_` on the next latent decode step. The instrumentation writes only on distributed rank 0, saves float32 CPU copies, and leaves the live GPU tensor unchanged.

The probe uses known naturally-triggered VStarBench position 0 and requires the generated token IDs to match the previously recorded baseline token IDs exactly. It passes only if generation is unchanged and exactly 10 finite latent vectors with one common hidden size are captured.

## Next Milestones
- [x] Pin Monet implementation and checkpoint.
- [x] Reproduce runtime and official inference path.
- [x] Verify latent token IDs and runner patch.
- [x] Bring up pinned VLMEvalKit.
- [x] Characterize natural latent triggering over all 191 examples.
- [x] Implement and validate exact latent-start suppression.
- [x] Run paired latent-off intervention over all 72 naturally-triggered examples.
- [x] Audit and repair answer parsing with option-aware rescoring.
- [x] Establish that paired correctness is neutral under the local robust scorer.
- [ ] Recover/document the supplementary API judge protocol if possible.
- [ ] Verify Step 12 exact latent-state tensor capture on one sample.
- [ ] Run no-training positive/evidence-preserving vs negative/evidence-destroying visual-view latent separability test.
- [ ] Start V0 only if the separability/utility gate is supported.

## Known Issues
- GPU server cannot directly access GitHub; use local staging/direct handoff.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.
- Monet evaluation README requires an API judge but does not identify the exact judge model in the evaluation section.

## Next Action
Run only the Step 12 single-sample tensor-capture engineering probe. Do not introduce positive/negative image interventions yet. If the probe preserves the baseline token sequence exactly and captures 10 finite hidden-state vectors, record that result and then design the `z(I)`, `z(I^+)`, `z(I^-)` separability experiment.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.