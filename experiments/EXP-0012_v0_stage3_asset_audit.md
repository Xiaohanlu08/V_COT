# EXP-0012 — V0 Stage-3 Source and Local Asset Audit

**Status:** COMPLETED  
**Purpose:** Determine whether the local environment already contains the source hooks, packages, SFT dataset, SFT checkpoints, and teacher-latent assets required for a minimal V0 Stage-3 extension.

## Source integrity
```text
actual_monet_sha: 08939998d3d643a73a316e349faa34f420429153
source_ready: true
CustomTrainerSFT_STAGE3: true
stage3_latent_forward: true
stage3_ce_alignment_objective: true
collate_fn_sft_stage3: true
student_pixel_values: true
student_alignment_poss: true
```

## Environment
```text
torch: 2.7.1
torchvision: 0.22.1
transformers: 4.54.0
trl: 0.15.2
accelerate: 1.15.0
datasets: 5.0.1
deepspeed: 0.19.6
qwen-vl-utils: 0.0.14
huggingface-hub: 0.36.2
```

## Local training assets
```text
Monet-SFT-125K dataset: absent
Monet-SFT-7B checkpoint: absent
precomputed teacher latents: absent
Stage1/2/3 training checkpoints: absent
final local Monet-7B inference checkpoint: present
```

Readiness:
```text
source_hook_audit_ready: true
official_stage3_recipe_locally_complete: false
continued_sft_v0_assets_minimum: false
```

## Conclusion
The code/runtime side is ready, but the local training assets are not. Do not reconstruct Stage1/2 or precompute official teacher latents yet. First inspect the metadata/schema of `Monet-SFT-125K` to determine whether its interleaved assistant/helper images can provide the scalable evidence signal needed by V0, then download only the assets required by the frozen V0 construction rule.
