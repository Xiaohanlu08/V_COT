# Current State

## Project Stage
Latent-state characterization / visual-evidence intervention pilot.

## Current Objective
Determine whether Monet's recurrent latent hidden states are selectively sensitive to task-relevant visual evidence and therefore suitable targets for selective supervision.

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

After option-aware rescoring of all 72 paired baseline/latent-off outputs:
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
These vectors are subsequently consumed through `self.inputs_embeds.index_copy_` as the next latent-step input embeddings. The live GPU tensor is unchanged; only rank-0 float32 CPU copies are saved.

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

The exact generated token sequence matched the previously recorded natural baseline. This verifies observation-only tensor capture for the tested sample.

## Official V*Bench Evidence Annotation — VERIFIED
The official annotation for VStarBench position 0 was fetched from `craigwu/vstar_bench`, file `direct_attributes/sa_4690.json`, and checked against the local image.

Verified fields:
```text
target_object: ['glove']
bbox: [[564, 142, 155, 157]]   # <x,y,w,h>
question: What is the material of the glove?
local image size: 2000 x 1500
annotation check: PASS
```

The official annotation options are semantic answer sentences and differ in ordering/format from the VLMEvalKit TSV options; the evidence pilot uses only the verified target object / bbox / question for visual intervention and keeps the existing VLMEvalKit prompt unchanged.

## Step 13 — Natural Three-View Evidence Pilot IMPLEMENTED, NOT YET VERIFIED
Implemented files:
- `scripts/13_vstar_evidence_pilot.py`
- `scripts/13_vstar_evidence_pilot.sh`

### View construction
For the first pilot, use the benchmark-provided target box and the V* reference implementation's target-patch scale `1.2`:
- `I`: original local VStarBench image.
- `I+`: target-centered `1.2x` V* patch with evidence preserved, resized back to the original `2000 x 1500` dimensions.
- `I-`: the exact same patch and resize transform as `I+`, but the annotated glove box is replaced by the RGB mean of its surrounding ring before resizing.

Thus `I+` and `I-` share geometry and image dimensions; their intended differential factor is the annotated target evidence.

### Natural-alignment gate
Step 13 intentionally does **not** force the latent token or replay a textual prefix yet. It first asks whether the three conditions naturally produce an already-aligned comparison under identical prompt/decoding settings.

The comparison is considered clean enough for an initial natural pilot only if all of the following hold:
```text
all three views naturally trigger exactly one latent segment
all three capture exactly 10 latent tensors
pre-latent generated token prefix matches original vs positive exactly
pre-latent generated token prefix matches original vs negative exactly
latent-start generated position is identical across all three views
```

If any condition fails, similarity values are exploratory only and the next step is fixed-prefix/fixed-trigger replay. If all conditions pass, compute step-aligned cosine similarities:
```text
S+ = mean_t cos(z_t(I), z_t(I+))
S- = mean_t cos(z_t(I), z_t(I-))
Delta_evidence = S+ - S-
```

The original view is also required to reproduce the saved natural baseline token sequence exactly under the instrumented runner.

## Formal Baseline Reproduction Caveat
Monet's README requests a supplementary API judge but does not identify the exact judge model/configuration in the evaluation section. Local option-aware rescoring is deterministic but is not the paper's under-specified API judge. This does not block the latent-state evidence experiment.

## Next Milestones
- [x] Pin Monet implementation and checkpoint.
- [x] Reproduce runtime and official inference path.
- [x] Characterize natural latent triggering over all 191 examples.
- [x] Validate exact latent-start suppression.
- [x] Re-score paired latent-off outputs robustly and close answer-level utility line as neutral.
- [x] Capture exact recurrent latent hidden-state tensors without changing generation.
- [x] Verify official V*Bench bbox/target annotation for pilot sample 0.
- [ ] Run Step 13 natural three-view evidence pilot.
- [ ] If natural prefix/trigger alignment fails, implement fixed-prefix / fixed-trigger latent replay and verify original-image latent reproduction.
- [ ] Compare `z(I)`, `z(I+)`, `z(I-)` under a mechanically aligned protocol.
- [ ] Expand to multiple annotated naturally-triggered samples only if the pilot protocol is valid.
- [ ] Start V0 only if the visual-evidence separability/utility gate is supported.

## Known Issues
- GPU server cannot directly access GitHub; use local staging/direct handoff.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.
- Monet evaluation README requires an API judge but does not identify the exact judge model in the evaluation section.

## Next Action
Run only the Step 13 natural three-view evidence pilot. Do not start V0 and do not implement forced replay unless Step 13 shows that the positive/negative views fail the strict natural-alignment gate.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
