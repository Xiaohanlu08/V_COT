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
Step 19 verified source hooks and the local training environment. DeepSpeed is installed and the pinned Monet source is intact, but the large SFT assets/checkpoints are not yet present locally. No package changes are required.

## EXP-0013 — Monet-SFT-125K Metadata / Schema Audit — COMPLETE
Step 20b audited the complete six-subset `train.json` metadata without downloading image archives or Stage-3 weight shards.

Global result:
```text
total rows: 125072
rows with user image: 125072 / 125072
rows with assistant/helper image: 125072 / 125072
rows with both user and assistant images: 125072 / 125072
rows with <observation>...</observation>: 124233 / 125072 = 99.3292%
```

Assistant/helper image multiplicity:
```text
1 image : 122072 samples
2 images: 303
3 images: 440
4 images: 480
5 images: 443
6 images: 529
7 images: 510
8 images: 294
11 images: 1
```

Subset structure:
```text
Visual_CoT              118561 rows; exactly 1 assistant image/sample
CogCoM                      567 rows; 1-11 assistant images/sample
ReFocus                     426 rows; almost always 1 assistant image
Zebra_CoT_count            2766 rows; 2-8 assistant images/sample
Zebra_CoT_visual_search    2687 rows; exactly 1 assistant image/sample
Zebra_CoT_geometry           65 rows; almost always 1 assistant image
```

Representative metadata confirms that assistant images are explicit intermediate visual evidence/transformation states paired with `<abs_vis_token></abs_vis_token>` reasoning steps: zoom/focus views, geometry/line-construction views, and sequential scene states are all represented depending on the source dataset.

Stage-3 checkpoint index reports total weight size `16,578,684,928` bytes; weight shards were not downloaded.

## Critical V0 Design Consequence
The schema audit confirms that a scalable positive visual-evidence source already exists for every Monet-SFT-125K sample: the assistant/helper image sequence. However, **simply adding those helper images as a positive view is not a new V0 mechanism**, because official Monet Stage 3 already uses helper-image-derived teacher latents to supervise the student latent trajectory.

Therefore V0 must add an explicitly **evidence-discriminative** component beyond existing Stage-3 positive alignment. The next design task is to define a negative/evidence-destroying counterpart that is causally meaningful and cannot be solved merely by sample identity or unrelated-image mismatch.

The helper-image structure is heterogeneous across datasets. A single universal negative operator must not be assumed before auditing actual image-pair transformations. In particular, multi-step CogCoM / Zebra-count helper sequences should not be silently treated as equivalent to single zoom/focus helpers.

## V0 Design Constraints Now Frozen
- Reuse Monet Stage 3 rather than creating a new architecture.
- Preserve the existing CE + teacher-latent alignment objective as the matched baseline component.
- Treat official helper images as the existing positive-evidence supervision source, not as the claimed novelty by itself.
- Add a negative/evidence-destroying contrastive term only after its construction rule is validated.
- Do not use random cross-sample helper images as the primary negative without an explicit ablation, because that can reduce to sample-identity discrimination rather than visual-evidence discrimination.
- Do not download Stage1/2 or precomputed teacher latents yet.
- Do not start full V0 training yet.

## Next Milestones
- [x] Validate target-specific latent sensitivity under outcome-blind multi-sample replication.
- [x] Audit local Stage-3 source hooks and environment.
- [x] Establish that local SFT data/checkpoints are currently absent.
- [x] Audit all Monet-SFT-125K train metadata and helper-image structure.
- [ ] Audit a small deterministic set of actual user/helper image pairs and classify the helper transformation types.
- [ ] Freeze the V0 negative/evidence-destroying construction rule.
- [ ] Download only the assets required by that frozen rule plus the published Stage-3 SFT checkpoint.
- [ ] Implement the minimal Stage-3 evidence-discriminative loss extension.
- [ ] Run a tiny deterministic smoke test, then a matched control-vs-V0 pilot.
- [ ] Move to V1/V2 only if V0 improves the predefined matched evaluation metric.

## Next Action
Do not write the V0 loss yet. First inspect a small deterministic, source-stratified set of actual user/helper image pairs to determine which transformation families are present (zoom/crop, annotation/drawing, sequential state, etc.) and whether a same-sample evidence-destroying negative can be generated without introducing a geometry or identity shortcut.