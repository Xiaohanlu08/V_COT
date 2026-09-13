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

## EXP-0007 — Fixed-Prefix / Fixed-Trigger Replay — MECHANICALLY VALID; ORIGINAL POSITIVE-NEGATIVE HYPOTHESIS NOT SUPPORTED ON SAMPLE 0
Step 14 replayed the exact 25 baseline generated tokens preceding natural latent start as explicit `prompt_token_ids`, used identical textual prompt IDs for all visual conditions, and forced only the first subsequent token to `151666=<abs_vis_token>`.

Mechanical alignment:
```text
prefix_generated_token_count: 25
prefix_decode_encode_roundtrip_exact: true
base_prompt_ids_identical_across_views: true
chat_template_text_identical_across_views: true
original: first_token=151666, segment=(0,10), latent_shape=(10,3584)
positive: first_token=151666, segment=(0,10), latent_shape=(10,3584)
negative: first_token=151666, segment=(0,10), latent_shape=(10,3584)
mechanical_alignment_pass: true
```

Original-image replay reproduced EXP-0005 closely:
```text
mean cosine: 0.9999062
min cosine: 0.9998059
mean relative L2: 0.0133799
valid_for_counterfactual_comparison: true
```

Counterfactual similarity:
```text
S_positive_mean: 0.9516485
S_negative_mean: 0.9566512
Delta_evidence = S_positive - S_negative: -0.0050027
relative L2 original-positive: 0.3006730
relative L2 original-negative: 0.2855321
```

Step-wise `Delta_evidence` was positive only at step 1 and negative at the remaining 9/10 steps. Mean positive-vs-negative cosine was approximately `0.9889295`, showing that the two crop-based interventions remain very similar to each other while both differ substantially more from the full original image.

### Scientific interpretation
The fixed-prefix/fixed-trigger method is mechanically validated. However, the initial single-sample hypothesis `sim(z(I), z(I+)) > sim(z(I), z(I-))` is **not supported** for position 0 under the crop-based intervention. The negative mean delta must not be reframed as a positive result.

The pattern suggests crop/resize geometry is a larger latent perturbation than target masking in this pilot: `I+` and `I-` share crop geometry and are highly similar to each other, while both are much farther from the full-image anchor. This motivates a cleaner target-specificity test that keeps the full-image geometry fixed.

## Step 15 — Full-Image Target Occlusion vs Matched Sham Occlusions PREPARED
Implemented:
- `scripts/15_vstar_target_occlusion_specificity.py`
- `scripts/15_vstar_target_occlusion_specificity.sh`

Protocol:
1. Keep the original `2000 x 1500` image geometry for every condition.
2. Neutral-mask only the annotated glove box for the target intervention.
3. Construct 32 deterministic same-size sham masks elsewhere in the same image, excluding the glove and one-box-width/height surrounding context.
4. Each mask is filled with its own surrounding-ring mean RGB, so target and sham interventions use the same operator.
5. Reuse the validated 25-token fixed prefix and one-time forced latent start from Step 14.
6. Revalidate the original replay against EXP-0005 before interpretation.
7. Measure latent perturbation as `1 - mean cosine(z_original, z_masked)` and compare the target mask against the empirical sham-mask distribution.

Primary target-specificity statistic:
```text
specificity = target cosine distance - median sham cosine distance
```
A positive value means occluding the annotated target perturbs the latent trajectory more than a typical matched same-size nuisance occlusion. With 32 sham controls, the smallest attainable one-sided empirical p-value is `1/33 ≈ 0.0303`. This remains a single-sample pilot, not a benchmark-level significance claim.

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
- [ ] Run Step 15 full-image target-vs-sham occlusion specificity pilot.
- [ ] Expand to multiple annotated naturally-triggered samples only if Step 15 supports a target-specific signal.
- [ ] Start V0 only if multi-sample evidence supports the visual-evidence gate.

## Known Issues
- GPU server cannot directly access GitHub; use artifact handoff / WinSCP.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.

## Next Action
Run only Step 15 target-vs-sham full-image occlusion specificity pilot. Do not train V0 yet.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
