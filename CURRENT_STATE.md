# Current State

## Project Stage
Latent-state characterization / visual-evidence intervention preparation.

## Current Objective
Determine whether Monet's recurrent latent hidden states are sensitive to task-relevant visual evidence and therefore suitable targets for selective supervision.

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

## Natural Latent Trigger Characterization — COMPLETE
Recorded as `EXP-0001` through `EXP-0003`.

Full VStarBench (`n=191`):
```text
triggered_samples: 72/191 = 0.3769633508 (~37.7%)
balanced_marker_samples: 191/191
multi_segment_samples: 0/191
all 72 latent segments have length 10
```

The original simple boxed-option parser was later shown to be unreliable for semantic answer text. Old parser-derived correctness values are retained only as historical diagnostics and are not used for scientific conclusions.

## Paired Latent-Suppression Ablation — COMPLETE
Recorded as `EXP-0004`.

The technically validated intervention blocks exactly `151666=<abs_vis_token>` while leaving all other model-vocabulary IDs available. After option-aware rescoring of all 72 paired baseline/latent-off outputs:
```text
baseline correct:   54/72 = 0.750000
latent-off correct: 54/72 = 0.750000
delta: 0.000000
correct -> correct: 50
correct -> wrong:    4
wrong   -> correct:  4
wrong   -> wrong:   14
McNemar exact two-sided p = 1.0
unresolved baseline: 0
unresolved latent-off: 2 (positions 0, 34)
```

Therefore no systematic answer-level correctness effect of blocking latent entry is detected at this scale. The earlier apparent latent-access benefit was a parser artifact.

## Exact Latent Tensor Capture — VERIFIED
Recorded as `EXP-0005`.

Step 12 instruments a temporary copy of the pinned official Monet vLLM runner exactly where Monet assigns:
```python
st["pending"] = last_token_h[i].detach()
```
These `pending` vectors are subsequently consumed through `self.inputs_embeds.index_copy_` as the next latent-step input embeddings. The live GPU tensor is unchanged; only rank-0 float32 CPU copies are saved.

Verified on naturally-triggered VStarBench position 0:
```text
generation_exact_match_baseline: true
num_latent_tensors: 10
hidden_size: 3584
dtype_saved: torch.float32
all_finite: true
mean_l2_norm: 288.1265869
min_l2_norm: 275.2210999
max_l2_norm: 293.0903931
adjacent_cosine_mean: 0.9219154
adjacent_cosine_min: 0.6129838
adjacent_cosine_max: 0.9985483
VSTAR_SINGLE_LATENT_TENSOR_PROBE_PASS=True
```

The exact generated token sequence matched the previously recorded natural baseline. This verifies that the tensor instrumentation is observation-only for the tested sample and that the captured `10 x 3584` sequence is the recurrent latent state actually fed by Monet between latent decode steps.

Saved result:
```text
results/latent_capture/vstar_pos0_latents.pt
results/latent_capture/vstar_pos0_latents_summary.json
logs/12_vstar_single_latent_tensor_probe.log
```

## Interpretation of Step 12
The high adjacent cosine similarity indicates a smooth recurrent trajectory overall, but the minimum adjacent cosine (`~0.613`) shows at least one comparatively large transition. These numbers are descriptive only; they do not establish visual grounding or usefulness.

The central scientific gate can now be tested directly: whether a task-evidence-preserving visual intervention keeps the latent trajectory closer to the original than an evidence-destroying intervention.

## Evidence Intervention Design Constraint
For a clean first test, the visual intervention should use benchmark-provided target annotations rather than a learned grounding model. The original V*Bench release documents per-sample `target_object` and `bbox` annotations in `<x,y,w,h>` format. The next step is to recover the official annotation for VStarBench position 0 (`direct_attributes/sa_4690`) and verify that it matches the local image before constructing matched positive/negative views.

Preferred first-pilot view construction after annotation verification:
- `I`: original image.
- `I+`: target-centered crop with context, target evidence preserved.
- `I-`: the exact same crop and dimensions as `I+`, but the annotated target box is neutral-masked.

This makes the positive and negative conditions share the same crop/context geometry, with the target evidence as the main differential factor.

For the actual latent comparison, text-prefix/trigger timing should also be controlled. The planned protocol is to replay the exact baseline pre-latent generated token prefix and force only the latent-start event at the same position for `I`, `I+`, and `I-`. The original-image replay must first reproduce the Step-12 latent tensors before intervention comparisons are considered valid.

## Formal Baseline Reproduction Caveat
Monet's README requests a supplementary API judge but does not identify the exact judge model/configuration in the evaluation section. Local option-aware rescoring is deterministic but is not the paper's under-specified API judge. This does not block the latent-state evidence experiment.

## Next Milestones
- [x] Pin Monet implementation and checkpoint.
- [x] Reproduce runtime and official inference path.
- [x] Characterize natural latent triggering over all 191 examples.
- [x] Validate exact latent-start suppression.
- [x] Re-score paired latent-off outputs robustly and close answer-level utility line as neutral.
- [x] Capture exact recurrent latent hidden-state tensors without changing generation.
- [ ] Recover and verify official V*Bench bbox/target annotation for pilot sample 0.
- [ ] Generate matched `I+` / `I-` evidence interventions for sample 0.
- [ ] Implement fixed-prefix / fixed-trigger latent replay and verify original-image latent reproduction.
- [ ] Compare `z(I)`, `z(I+)`, `z(I-)` with step-aligned and sequence-level similarity metrics.
- [ ] Expand to multiple annotated naturally-triggered samples only if the pilot protocol is mechanically valid.
- [ ] Start V0 only if the visual-evidence separability/utility gate is supported.

## Known Issues
- GPU server cannot directly access GitHub; use local staging/direct handoff.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.
- Monet evaluation README requires an API judge but does not identify the exact judge model in the evaluation section.

## Next Action
Do not modify training or run V0. First fetch/verify the official V*Bench annotation for VStarBench position 0 (`sa_4690`) and confirm its target object and bounding box against the local sample. Then construct matched positive/negative evidence views and proceed to fixed-prefix latent replay.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.