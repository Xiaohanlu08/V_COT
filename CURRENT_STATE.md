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
Step 21e successfully inspected the same preselected three examples per source subset using byte-range extraction from the current `images.zip` archives. No full archive was downloaded. The transport path is mechanically validated and reusable.

Observed transformation families:
- `Visual_CoT`: sampled helpers are much smaller than source images, consistent with localized crop/zoom evidence.
- `CogCoM`: same-canvas annotation/construction-style views.
- `ReFocus`: same-canvas focus/highlight transforms.
- `Zebra_CoT_count`: multi-step state/view transformations.
- `Zebra_CoT_visual_search`: internally heterogeneous (small crops plus rendered/boxed views).
- `Zebra_CoT_geometry`: geometry construction/decomposition views.

This rejects a universal negative operator across all six subsets.

## V0 Design Direction — NARROWED, NOT YET FROZEN
The cleanest first V0 family is `Visual_CoT`, because it is dominant (~94.8% of SFT data) and appears to provide localized crop/zoom evidence. Candidate construction:
1. recover the helper crop location in the same source image;
2. use the official helper crop as positive evidence;
3. sample a same-source, same-size/aspect-ratio sham crop outside the recovered positive region as the negative;
4. compare latent similarity to positive vs negative evidence.

This remains provisional until crop recoverability is quantitatively validated.

## Step 22 — Quantitative Visual_CoT Crop Audit — INVALID IMPLEMENTATION RUN
The exact outcome-blind cohort was frozen at 64 rows with seed `20260913`:
```text
selection_sha256 = b375ac07747c17525187cafc5c842fed1a20a60d1174cc927d5c3f9bc9f89063
```
All 64 source/helper image pairs were successfully extracted before matching failed.

The partial matching output is **not scientifically interpretable**. Two implementation defects were exposed:
1. float32 NCC denominator cancellation on near-constant windows produced mathematically impossible correlation magnitudes greater than 1 (examples included `45.9486`, `1047.7974`, `158.8655`), proving the original NCC implementation was numerically invalid;
2. a wrong-source control later returned `None`, and logging crashed on `NoneType.__format__` before all 64 samples were evaluated.

Therefore none of the partial Step-22 matching values may be used to judge the crop-family hypothesis.

The frozen scientific thresholds are unchanged:
```text
strong sample:
  same-source NCC >= 0.90
  same-minus-wrong NCC margin >= 0.05

family success gate:
  all 64 pairs valid
  strong fraction >= 0.90
  median same-source NCC >= 0.95
  median margin >= 0.10
```

Step 22b is an implementation-only rerun:
- exact same 64 rows; no re-sampling;
- all 64 already-extracted image pairs reused offline;
- float64 NCC accumulation;
- near-constant templates/windows rejected instead of epsilon division;
- mathematical NCC bound `[-1,1]` enforced;
- wrong-source control remains deterministic cyclic but skips only sources that cannot geometrically fit the helper under any frozen scale; this choice uses dimensions only, never match outcomes;
- no scientific threshold is relaxed after seeing the failed partial output.

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
- [ ] Complete the robust offline rerun of the frozen 64-sample Visual_CoT crop-recoverability audit.
- [ ] If crop recoverability passes the unchanged gate, freeze the same-source matched-sham negative construction.
- [ ] Download only the published checkpoint/data assets required for the resulting V0 pilot.
- [ ] Implement the minimal evidence-discriminative loss extension.
- [ ] Run a tiny deterministic smoke test, then a matched control-vs-V0 pilot.
- [ ] Move to V1/V2 only if V0 improves the predefined matched evaluation metric.

## Next Action
Run Step 22b offline on the exact 64 image pairs already extracted by Step 22. Do not re-download, re-sample, or alter the frozen thresholds based on the invalid partial Step-22 outputs.