# EXP-0019 — Stage-3 Runtime Contract Probe

**Status:** COMPLETED

**Purpose:** Establish the actual Stage-3 runtime tensor contract and memory envelope before freezing the V0 evidence loss or downloading additional SFT checkpoints.

**Assets:** Existing local `models/Monet-7B`; pinned Monet source `08939998d3d643a73a316e349faa34f420429153`; Visual_CoT row 0; Step-22 source/helper images; frozen Step-24 neutral-occluded negative. No network download or package modification.

**Step 26 partial result:** Latent forward succeeded, but the synthetic hidden-state diagnostic omitted `alignment_poss` and therefore aborted in pinned Monet's `output_hidden_states=True` path. This was a probe-script plumbing error, not a model/source failure.

**Step 26b correction:** Passed the exact Stage-3 `alignment_poss`, consumed Monet's actual per-sample aligned hidden-state return structure, and completed the mechanics probe.

## Verified Runtime Contract
```text
positive input_ids shape: [1,334]
negative input_ids shape: [1,334]
token IDs identical: true
alignment positions: [308,309,310,311,312,313,314,315]
image grid identical: true
positive ce_patch_vec: list[Tensor(8,3584)] bf16
negative ce_patch_vec: list[Tensor(8,3584)] bf16
positive ce_patch_pos: [308..315]
negative ce_patch_pos: [308..315]
latent count = alignment-position count = 8
positive aligned hidden shape: [29,8,3584]
negative aligned hidden shape: [29,8,3584]
```

## Sequential Inference-Only Memory
```text
model baseline allocated: 15.487451 GiB
positive latent peak allocated: 15.563718 GiB
negative latent peak allocated: 15.580688 GiB
positive hidden-forward peak allocated: 15.645299 GiB
alignment functional-probe peak allocated: 15.685331 GiB
```
These numbers do not prove train-time feasibility because no backward graph or optimizer states were present.

## Synthetic Self-Target Diagnostics — Mechanics Only
Using the original branch's own aligned hidden tensor as a synthetic target:
```text
manual official/default-dim positive distance: ~4.98e-09
manual official/default-dim negative distance: ~0.0776314
manual official gap (negative-positive): ~0.0776314
explicit dim=-1 positive distance: ~-1.75e-08
explicit dim=-1 negative distance: ~0.0197890
explicit dim=-1 gap (negative-positive): ~0.0197890
official functional positive alignment: ~-8.82e-05
official functional negative alignment: ~0.0776367
```
The tiny negative self-distances are floating-point effects. These values are not official Stage-2 teacher measurements and must not be used to select a margin or claim evidence discrimination.

## Conclusion
**KEEP.** The runtime tensor space is resolved. A Visual_CoT sample with `latent_size=8` yields eight 3584-D recurrent latent vectors and an all-layer aligned hidden tensor `[29,8,3584]`. The original and frozen occluded branches are sequence- and alignment-compatible.

The public Stage-3 checkpoint is not needed for further shape inspection. The next justified asset is only the public `Monet-SFT-7B/stage2` checkpoint so that a real official teacher target can be computed for a tiny fixed sample. Do not rebuild Stage1 or full teacher caches yet.