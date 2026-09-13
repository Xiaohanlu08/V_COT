# Current State

## Project Stage
V0 preparation: evidence-supported latent contrastive supervision.

## Current Objective
Translate the confirmed target-specific latent sensitivity signal into the minimal V0 SFT intervention defined in `PROJECT_GOAL.md`, without changing Monet's architecture or RL pipeline.

## Pinned Baseline
- Monet source: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Current reproduced inference checkpoint: local `models/Monet-7B`
- VLMEvalKit snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Protected runtime: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, transformers 4.54.0, trl 0.15.2, vllm 0.10.0.

## Evidence Gate — PASSED
The outcome-blind confirmatory direct-attribute cohort was frozen at positions:
`[6, 11, 22, 23, 27, 53, 57, 59, 67, 71, 102, 112]`
with SHA-256:
`f65ebdbefc8a73276877f4096d5cb8897339447c91a31dc68322a811dda487e4`.

EXP-0011 confirmatory result under `protocols/CONFIRMATORY_OCCLUSION_12.md`:
```text
mechanically valid: 12/12
positive specificity: 10/12
negative specificity: 2/12
specificity mean:   +0.0008347084
specificity median: +0.0004564524
exact one-sided sign-test p: 0.019287109375
exact sign-flip mean p:     0.0009765625
primary_hypothesis_supported: true
```

Supported claim: under the tested direct-attribute/full-image neutral-occlusion protocol, masking task-relevant target evidence perturbs Monet's aligned recurrent latent trajectory more than matched nuisance occlusions across samples.

Do **not** claim that this proves answer-level utility or universal visual grounding. The earlier latent-off ablation remained null after robust rescoring (`54/72` vs `54/72`, McNemar `p=1.0`).

## V0 Gate Decision
The evidence gate for beginning V0 is **OPEN**. V0 remains SFT-only and architecture-preserving. VStarBench confirmatory samples are probing/evaluation evidence only and must not be used as V0 training data.

## Monet Stage-3 Integration Facts — VERIFIED
Official Stage 3 already provides the correct hook points:
- `collate_fn_sft_stage3` builds student inputs from user/question images after removing auxiliary assistant images;
- `CustomTrainerSFT_STAGE3` performs a latent forward and exposes `student_outputs_latent.ce_patch_vec`;
- existing Stage-3 objective is `student_ce_loss + alignment_weight * alignment_loss`;
- therefore V0 should extend Stage 3 rather than introduce a new architecture.

## EXP-0012 — Local V0 Stage-3 Asset Audit — COMPLETE
Step 19 ran without model loading, GPU inference, or package changes.

Source/hook integrity:
```text
actual Monet SHA: 08939998d3d643a73a316e349faa34f420429153
source_ready: true
CustomTrainerSFT_STAGE3: true
stage3_latent_forward: true
stage3_ce_alignment_objective: true
collate_fn_sft_stage3: true
student_pixel_values: true
student_alignment_poss: true
```

Current environment:
```text
torch 2.7.1
transformers 4.54.0
trl 0.15.2
accelerate 1.15.0
datasets 5.0.1
deepspeed 0.19.6
qwen-vl-utils 0.0.14
huggingface-hub 0.36.2
```

Training assets currently absent:
```text
Monet-SFT-125K: absent
Monet-SFT-7B: absent
precomputed teacher latents: absent
Stage1/2/3 training checkpoints: absent
```

Available:
```text
local final Monet-7B inference checkpoint: present
source_hook_audit_ready: true
official_stage3_recipe_locally_complete: false
continued_sft_v0_assets_minimum: false
```

## Immediate Design Consequence
Do **not** download every training asset or reconstruct all official SFT stages yet. The next question is whether `Monet-SFT-125K` itself provides a scalable positive-evidence signal through its interleaved assistant/helper images, which Stage 3 explicitly removes from the student input. If so, V0 can likely start from the published Stage-3 SFT checkpoint and use those helper images as evidence supervision, avoiding unnecessary Stage1/2 reconstruction and teacher-latent precomputation for the first pilot.

## Next Milestones
- [x] Validate target-specific latent sensitivity under outcome-blind multi-sample replication.
- [x] Audit local Stage-3 source hooks and environment.
- [x] Establish that local SFT data/checkpoints are currently absent.
- [ ] Download only Monet-SFT-125K metadata/train JSONs and Stage-3 checkpoint metadata; audit helper-image structure before downloading multi-GB assets.
- [ ] Freeze the V0 positive/negative evidence construction rule before training-data generation.
- [ ] Download only the assets required by that frozen rule.
- [ ] Implement the minimal Stage-3 V0 loss extension.
- [ ] Run a tiny deterministic smoke test, then a matched control-vs-V0 pilot.
- [ ] Move to V1/V2 only if V0 improves the predefined matched evaluation metric.

## Next Action
Run a metadata-only Monet-SFT-125K schema audit. Do not start full V0 training and do not download Stage1/2/teacher-latent assets yet.
