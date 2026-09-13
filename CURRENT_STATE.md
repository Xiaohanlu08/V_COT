# Current State

## Project Stage
Latent-state characterization / controlled visual-evidence intervention.

## Current Objective
Determine whether Monet's recurrent latent hidden states are selectively sensitive to task-relevant visual evidence under a mechanically aligned counterfactual protocol.

## Current Branch
`main`

## Pinned Baseline
- Monet: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Checkpoint: local `models/Monet-7B`
- VLMEvalKit reproducibility snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Runtime remains protected: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, transformers 4.54.0, trl 0.15.2, vllm 0.10.0.

## Completed Baseline / Mechanics
- Natural VStarBench latent triggering: `72/191 = 37.7%`; all markers balanced; no multi-segment samples; all latent segments length 10.
- Paired latent-start suppression on all 72 naturally triggered samples: exact token exclusion verified 72/72.
- After deterministic option-aware rescoring, baseline and latent-off are both `54/72 = 0.75`; transitions `CC=50, CW=4, WC=4, WW=14`; McNemar exact `p=1.0`. No detectable answer-level correctness effect.
- Exact recurrent latent tensor capture verified on VStarBench position 0. Observation-only instrumentation preserved the baseline token sequence exactly and captured `10 x 3584` finite vectors from `st["pending"]`.

## EXP-0005 Tensor-Capture Reference
VStarBench position 0:
```text
generation_exact_match_baseline: true
num_latent_tensors: 10
hidden_size: 3584
mean_l2_norm: 288.1265869
adjacent_cosine_mean: 0.9219154
adjacent_cosine_min: 0.6129838
adjacent_cosine_max: 0.9985483
```
Reference tensor:
`results/latent_capture/vstar_pos0_latents.pt`

## Official V*Bench Evidence Annotation — VERIFIED
Official `craigwu/vstar_bench/direct_attributes/sa_4690.json` matches local VStarBench position 0:
```text
target_object: ['glove']
bbox: [[564, 142, 155, 157]]   # <x,y,w,h>
question: What is the material of the glove?
local image size: 2000 x 1500
```

## EXP-0006 — Natural Three-View Evidence Pilot — ALIGNMENT FAILED
Step 13 used the original image, a V* target crop (`patch_scale=1.2`), and the same crop with the target neutral-masked.

Natural-generation result:
```text
original: segment [25,35], 10 latent tensors
positive crop: no latent segment
negative masked crop: segment [16,26], 10 latent tensors
all_three_naturally_triggered: false
same_pre_latent_prefix: false
same_latent_start_position: false
aligned_natural_comparison_ready: false
```

This shows the visual intervention changes Monet's latent-entry policy, but it does not permit direct latent-content comparison.

## EXP-0007 — Fixed-Prefix / Fixed-Trigger Crop Replay — HYPOTHESIS NOT SUPPORTED ON SAMPLE 0
Step 14 replayed the exact 25 baseline generated tokens preceding natural latent start as explicit `prompt_token_ids`, used identical textual prompt IDs for all visual conditions, and forced only the first subsequent token to `151666=<abs_vis_token>`.

Mechanical alignment passed for all three views, and original-image replay reproduced EXP-0005 closely:
```text
mean cosine: 0.9999062
min cosine: 0.9998059
mean relative L2: 0.0133799
valid_for_counterfactual_comparison: true
```

Counterfactual crop result:
```text
S_positive_mean: 0.9516485
S_negative_mean: 0.9566512
Delta_evidence = S_positive - S_negative: -0.0050027
relative L2 original-positive: 0.3006730
relative L2 original-negative: 0.2855321
```

The crop-based hypothesis `sim(z(I),z(I+)) > sim(z(I),z(I-))` was not supported. The pattern indicated crop/resize geometry was a larger perturbation than target masking, motivating a full-image target-specificity design.

## EXP-0008 — Full-Image Target Occlusion Specificity — PROMISING SINGLE-SAMPLE SIGNAL
Step 15 kept the full `2000 x 1500` image geometry fixed and compared:
- original image;
- neutral mask over the annotated glove bbox;
- 32 deterministic same-size sham masks elsewhere in the same image.

All conditions used the validated fixed 25-token prefix and one-time forced latent start. Original replay again passed the EXP-0005 reproduction gate.

Result:
```text
original_replay_valid: true
target_mean_cosine: 0.9989594221
target_cosine_distance: 0.0010405779
target_mean_relative_l2: 0.0439559296
control_cosine_distance_median: 0.0004835725
control_cosine_distance_mean: 0.0005317628
specificity_score: +0.0005570054
target_distance / median-control distance: ~2.15x
target_distance_percentile_among_controls: 96.875
num_controls_with_distance_ge_target: 1/32
empirical_one_sided_p: 0.0606061
VSTAR_TARGET_OCCLUSION_SPECIFICITY_PASS=True
```

### Scientific interpretation
For VStarBench position 0, occluding the benchmark-annotated glove perturbs the recurrent latent trajectory about `2.15x` more than the median matched sham occlusion. The target mask lies at the `96.875`th percentile of the 32-control distribution, with only one control producing an equal-or-larger perturbation.

This is **promising target-specific latent sensitivity**, but it is not yet benchmark-level evidence and the single-sample empirical p-value (`0.0606`) is above the conventional `0.05` threshold. Do not claim statistical significance or visual grounding from this one sample. Increasing only the number of sham masks on the same sample would refine its within-image null but would not establish generalization; the next important step is replication across multiple naturally-triggered annotated samples.

## Formal Evaluation Caveat
Monet's README requests a supplementary API judge but does not specify the exact judge model/configuration. Local option-aware rescoring is deterministic but is not the paper's under-specified API judge. This does not block latent-state evidence experiments.

## Next Milestones
- [x] Pin Monet / runtime / benchmark path.
- [x] Characterize natural latent triggering over all 191 VStarBench samples.
- [x] Validate exact latent-start suppression and robust local answer rescoring.
- [x] Capture exact recurrent latent hidden tensors without changing generation.
- [x] Verify official target/bbox annotation for position 0.
- [x] Run natural three-view evidence pilot and identify alignment failure.
- [x] Validate fixed-prefix / fixed-trigger replay and original-image tensor reproduction.
- [x] Establish that the crop-based `I+` vs `I-` hypothesis is not supported on sample 0.
- [x] Run full-image target-vs-sham occlusion specificity pilot on sample 0.
- [ ] Audit metadata/annotation mapping for the 72 naturally-triggered samples.
- [ ] Define a preregistered multi-sample subset and run target-vs-sham specificity with the same operator.
- [ ] Aggregate specificity across samples with paired/nonparametric statistics and category stratification.
- [ ] Start V0 only if multi-sample evidence supports the visual-evidence gate.

## Known Issues
- GPU server cannot directly access GitHub; use artifact handoff / WinSCP.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.

## Next Action
Do not train V0 and do not spend the next step merely increasing the sham count for position 0. First audit the VLMEvalKit metadata for naturally-triggered samples so each position can be mapped reproducibly to its official V*Bench annotation (`target_object`, `bbox`, question, source image). Then freeze a small multi-sample replication subset before running additional GPU inference.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
