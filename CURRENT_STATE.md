# Current State

## Project Stage
V0 preparation: evidence-discriminative latent supervision.

## Current Objective
Turn the confirmed target-specific latent-sensitivity signal into the smallest architecture-preserving SFT experiment that tests whether same-sample visual-evidence discrimination can make Monet's latent states more useful.

## Pinned Baseline
- Monet source: `NOVAglow646/Monet@08939998d3d643a73a316e349faa34f420429153`
- Current reproduced inference checkpoint: local `models/Monet-7B`
- VLMEvalKit snapshot: `open-compass/VLMEvalKit@1e2b2f9934cd5ea05e54b8706ab40b09ec3d3ae3`
- Protected runtime unchanged: Python 3.10.21, torch 2.7.1+cu126, torchvision 0.22.1+cu126, transformers 4.54.0, trl 0.15.2, vllm 0.10.0.

## Evidence Gate — PASSED
The outcome-blind 12-sample direct-attribute confirmatory cohort passed the frozen target-vs-sham occlusion gate:
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
Supported claim: under the tested direct-attribute/full-image neutral-occlusion protocol, masking task-relevant evidence perturbs aligned recurrent Monet latent states more than matched nuisance occlusions. Do not upgrade this to answer-level utility or universal grounding; latent-start suppression remained null after robust rescoring (`54/72` vs `54/72`, McNemar `p=1.0`).

## V0 Gate Decision
The evidence gate for V0 is OPEN. V0 remains SFT-only, architecture-preserving, and must not use the VStarBench confirmatory cohort as training data.

## Monet Stage-3 Integration Facts — VERIFIED
Official Stage 3 already:
- removes assistant/helper images from the student input;
- exposes student latent states through `student_outputs_latent.ce_patch_vec`;
- uses helper-image-derived teacher latent supervision;
- trains with `student_ce_loss + alignment_weight * alignment_loss`.
The official Stage-3 recipe precomputes teacher latents with the SFT Stage-2 model, then trains the Stage-3 student. Therefore simply reusing helper images as a positive view is not itself a new mechanism.

## EXP-0012 — Local V0 Stage-3 Asset Audit — COMPLETE
Pinned source hooks and the local training environment are ready. DeepSpeed is installed. Large SFT assets/checkpoints are not yet local. No package changes are required.

## EXP-0013 — Monet-SFT-125K Metadata / Schema Audit — COMPLETE
All `125072/125072` examples contain a user image and at least one assistant/helper image. `124233/125072 = 99.3292%` contain `<observation>...</observation>` tags.

Helper multiplicity:
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
Subset sizes:
```text
Visual_CoT              118561
CogCoM                      567
ReFocus                     426
Zebra_CoT_count            2766
Zebra_CoT_visual_search    2687
Zebra_CoT_geometry           65
```
Visual_CoT alone is `118561/125072 = 94.7942%` of the official SFT data.

## EXP-0014 — Deterministic Actual Helper-Pair Audit — COMPLETE
Step 21e successfully inspected the same preselected three examples per source subset using byte-range extraction from the current `images.zip` archives. No full archive was downloaded. The transport path is now mechanically validated and reusable.

Selected rows:
```text
Visual_CoT:           [0, 6309, 24820]
CogCoM:               [0, 497, 565]
ReFocus:              [0, 66, 225]
Zebra_CoT_count:      [0, 37, 2251]
Zebra_CoT_visual_search: [0, 342, 2415]
Zebra_CoT_geometry:   [0, 42, 62]
```

Observed transformation families from image dimensions plus the paired reasoning instructions:
- `Visual_CoT`: all three helpers are much smaller than the source image (`114x161` vs `375x500`; `209x262` vs `281x500`; `61x70` vs `462x308`), consistent with localized crop/zoom evidence.
- `CogCoM`: helpers preserve the source canvas size and the text explicitly describes drawing line segments / locating chart regions; these are annotation/construction-style intermediate views rather than simple crops.
- `ReFocus`: helper and source sizes are identical and the text describes focusing/highlighting specific bars or chart regions; these are same-canvas focus/highlight transforms.
- `Zebra_CoT_count`: 4-8 same-size helper frames are used for viewpoint changes and sequential scene/state tracking; these are multi-step state/view transformations.
- `Zebra_CoT_visual_search`: two sampled helpers are small localized views (`198x116`, `206x176`) while another is a larger rendered/boxed view (`1232x1848` from `683x1024` source); this subset is itself heterogeneous.
- `Zebra_CoT_geometry`: helper dimensions remain close to the original canvas and the reasoning is geometry construction/decomposition; these should not be treated as crop positives.

This audit rejects a universal helper-image negative operator across all six subsets. The same geometry-preserving negative rule cannot be assumed valid for crop, annotation, highlight, sequential-state, and geometry-construction helpers.

## V0 Design Direction — NARROWED, NOT YET FROZEN
The cleanest first V0 family is `Visual_CoT`, because it is both dominant (~94.8% of SFT data) and appears to provide localized crop/zoom evidence. A scientifically clean candidate construction is:
1. recover the helper crop's location in the same source image;
2. use the official helper crop as positive evidence;
3. sample a same-source, same-size/aspect-ratio sham crop outside the recovered positive region as the negative;
4. compare latent similarity to positive vs negative evidence, avoiding cross-sample identity shortcuts and avoiding the crop/full-image geometry confound seen in EXP-0007.

This rule is still provisional. Three Visual_CoT examples are insufficient to assume that all 118561 helpers are recoverable crops.

## V0 Design Constraints Now Frozen
- Reuse Monet; do not add a new architecture.
- Keep V0 SFT-only; do not touch VLPO yet.
- Do not claim helper images themselves as the novelty; official Stage 3 already uses positive helper supervision.
- The novel V0 component must be evidence-discriminative.
- Do not use random cross-sample images as the primary negative.
- Do not mix helper transformation families under one negative operator without validation.
- Do not reconstruct Stage1/2 or download all training assets before the V0 data rule is frozen.
- Do not start full V0 training yet.

## Next Milestones
- [x] Validate target-specific latent sensitivity under outcome-blind multi-sample replication.
- [x] Audit local Stage-3 hooks/environment.
- [x] Audit complete Monet-SFT-125K metadata/helper structure.
- [x] Audit deterministic actual image pairs across all six source subsets.
- [ ] Quantitatively test Visual_CoT helper-to-source crop recoverability on a substantially larger deterministic sample.
- [ ] If crop recoverability is high, freeze the same-source matched-sham negative construction.
- [ ] Download only the published checkpoint/data assets required for the resulting V0 pilot.
- [ ] Implement the minimal evidence-discriminative loss extension.
- [ ] Run a tiny deterministic smoke test, then a matched control-vs-V0 pilot.
- [ ] Move to V1/V2 only if V0 improves the predefined matched evaluation metric.

## Next Action
Run a deterministic quantitative Visual_CoT crop-recoverability audit before writing the V0 loss. The audit should use many more than three examples and should measure whether each helper can be localized back into its own source image with a strong image-match score. Do not use answer accuracy or any downstream V0 outcome to select the examples.