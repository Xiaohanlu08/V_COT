# EXP-0027 — Stage-3 ZeRO-2 Model-Level Backward/Memory Smoke

## Status
ENGINEERING BLOCKER / SCIENTIFIC STATUS UNCHANGED

## Purpose
Test whether the verified Stage-1 Stage-3 student path and the frozen D015 two-branch evidence graph can complete forward, backward, and one optimizer step on the available RTX 3090 system under the official-style ZeRO-2 distributed configuration.

## Fixed engineering setup
- Student: verified public `models/Monet-SFT-7B-stage1`
- Fixed engineering row: Visual_CoT row `5345`
- `latent_size=8`
- `sft_stage3_img_tokens=2000`
- `ce_emphasize_factor=4.0`
- `alignment_weight=2.0`
- 8 GPUs, RTX 3090 class, ~23.57 GiB each
- DeepSpeed ZeRO stage 2, no CPU parameter/optimizer offload
- Baseline objective unchanged: CE + official alignment
- V0 adds the frozen D015 two-branch counterfactual evidence graph
- V0 engineering evidence weight `1.0` only to exercise the graph; it is not a lambda candidate

## Result
Both runs failed from true CUDA capacity exhaustion before backward.

### Baseline
The baseline completed the positive latent forward but OOMed in the positive second Stage-3 forward (`pos_out`). Representative rank-0 error:
```text
CUDA out of memory. Tried to allocate 12.00 MiB.
GPU capacity: 23.57 GiB
free: ~19 MiB
PyTorch allocated: ~23.09 GiB
reserved but unallocated: ~21.46 MiB
```

### V0
The V0 path OOMed earlier, while attempting the negative latent forward (`neg_latent`) with the positive latent graph still resident. Representative error:
```text
CUDA out of memory. Tried to allocate 20.00 MiB.
GPU capacity: 23.57 GiB
free: ~21 MiB
PyTorch allocated: ~23.09 GiB
reserved but unallocated: ~17.32 MiB
```

The same behavior was observed across multiple ranks.

## Interpretation
This is a real per-rank memory-capacity blocker, not primarily allocator fragmentation: almost the entire 24 GiB device is already allocated, and the unallocated reserved pool is only on the order of tens of MiB. The shell already enabled `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so allocator tuning is not an adequate fix.

Because the matched baseline itself OOMs under ZeRO-2, this result must **not** be interpreted as failure of the D015 evidence metric or as evidence-specific infeasibility.

Scientific results remain unchanged:
- EXP-0025 counterfactual-delta metric gate: PASSED
- EXP-0026 representation-gradient contract: PASSED
- `lambda_evidence`: still unfrozen

## Next engineering discriminator
Before changing image resolution or the scientific graph, test the exact same baseline/V0 computation under ZeRO-3 with no CPU offload. This changes only distributed memory partitioning and preserves the 2000-token image cap, model, data, objectives, and frozen D015 loss.

If ZeRO-3 still fails, then evaluate a separately documented hardware/input-resolution adaptation rather than modifying the scientific metric.