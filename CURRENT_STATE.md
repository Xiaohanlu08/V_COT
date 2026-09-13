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
Step 13 constructed:
- `I`: original image.
- `I+`: V* target-centered patch with `patch_scale=1.2`, resized to `2000 x 1500`.
- `I-`: exact same crop/resize as `I+`, but the glove bbox was replaced by the surrounding-ring mean RGB.

Verified geometry:
```text
crop_box_xyxy: [548, 126, 734, 314]
crop_size_before_resize: [186, 188]
local_target_bbox_xyxy: [16, 16, 171, 173]
neutral_fill_rgb: [137, 128, 130]
```

Natural-generation result:
```text
original: segment [25,35], 10 latent tensors, 99 generated tokens
positive: no latent segment, 0 latent tensors, 89 generated tokens
negative: segment [16,26], 10 latent tensors, 100 generated tokens
original_generation_exact_match_baseline: true
all_three_naturally_triggered: false
all_three_have_10_latents: false
same_pre_latent_prefix_original_vs_positive: false
same_pre_latent_prefix_original_vs_negative: false
same_latent_start_position: false
aligned_natural_comparison_ready: false
VSTAR_EVIDENCE_PILOT_CAPTURE_PASS=True
```

The positive view did not naturally enter latent mode, while the negative view entered latent mode earlier. Therefore no `z(I)` / `z(I+)` / `z(I-)` similarity from Step 13 is scientifically interpretable. The result itself is useful evidence that visual intervention changes the model's latent-entry policy, but it does not answer whether the latent state is selectively grounded in target evidence.

## Step 14 — Fixed-Prefix / Fixed-Trigger Replay PREPARED
Implemented:
- `scripts/14_make_fixed_trigger_runner.py`
- `scripts/14_vstar_fixed_prefix_evidence_replay.py`
- `scripts/14_vstar_fixed_prefix_evidence_replay.sh`

Protocol:
1. Take the exact 25 generated token IDs preceding the natural latent-start token in the recorded baseline for position 0.
2. Append those IDs directly as `prompt_token_ids`; do not rely on decoded-text retokenization.
3. Use identical base prompt token IDs for `I`, `I+`, and `I-` while changing only the image payload.
4. A temporary Monet runner patch overrides only the first sampled token of each replay request with `151666=<abs_vis_token>`; all subsequent sampling is untouched.
5. Capture the same recurrent `st["pending"]` tensors as EXP-0005.
6. Require all three replays to produce a latent segment `[0,10]` with `10 x 3584` tensors.
7. Before interpreting visual comparisons, require the original-image fixed replay to reproduce the EXP-0005 trajectory with: mean cosine >= 0.999, minimum cosine >= 0.995, and mean relative L2 <= 0.05.
8. Only if that replay gate passes, interpret step-aligned `S+`, `S-`, and `Delta_evidence = S+ - S-` as a single-sample counterfactual pilot.

## Formal Evaluation Caveat
Monet's README requests a supplementary API judge but does not specify the exact judge model/configuration. Local option-aware rescoring is deterministic but is not the paper's under-specified API judge. This does not block latent-state evidence experiments.

## Next Milestones
- [x] Pin Monet / runtime / benchmark path.
- [x] Characterize natural latent triggering over all 191 VStarBench samples.
- [x] Validate exact latent-start suppression and robust local answer rescoring.
- [x] Capture exact recurrent latent hidden tensors without changing generation.
- [x] Verify official target/bbox annotation for position 0.
- [x] Run natural three-view evidence pilot and identify alignment failure.
- [ ] Run Step 14 fixed-prefix / fixed-trigger replay.
- [ ] Verify original-image replay against EXP-0005 tensor trajectory.
- [ ] If replay gate passes, inspect `S+`, `S-`, and `Delta_evidence` for the pilot.
- [ ] Expand to multiple annotated naturally-triggered samples only after the pilot protocol is mechanically valid.
- [ ] Start V0 only if multi-sample evidence supports the visual-evidence gate.

## Known Issues
- GPU server cannot directly access GitHub; use artifact handoff / WinSCP.
- VLMEvalKit was transferred as an archive and has no local `.git` metadata.
- Windows-to-Linux transfers can introduce CRLF.
- vLLM shutdown may emit non-fatal NCCL/resource-tracker warnings.

## Next Action
Run only Step 14 fixed-prefix / fixed-trigger replay. Do not train V0 yet.

## Update Rule
After every verified step, update this file with current state, blockers, and next action. Scientific goals belong in `PROJECT_GOAL.md`, design decisions in `DECISIONS.md`, and numerical experiment records in `EXPERIMENTS.md`.
