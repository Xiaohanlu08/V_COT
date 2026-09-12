# Monet Baseline Specification

## Upstream
- Repository: `https://github.com/NOVAglow646/Monet.git`
- Pinned upstream commit: `08939998d3d643a73a316e349faa34f420429153`
- Commit date: 2026-03-19
- Commit message: `fix inference example`

This exact commit is the project baseline source until a new numbered decision in `DECISIONS.md` explicitly supersedes it.

## Official model/checkpoint used first
- Primary baseline checkpoint: `NOVAglow646/Monet-7B`
- Initial task: reproduce official inference before any training or code modification.
- Initial latent size: `LATENT_SIZE=10`, matching the latest upstream inference instructions at the pinned commit.

## Upstream implementation paths of interest
- SFT model: `monet_qwen_model/modeling_qwen2_5_vl_monet.py`
- RL Transformers implementation: `RL/monet_models/transformers`
- RL vLLM implementation: `RL/monet_models/vllm`
- Inference vLLM implementation: `inference/vllm/monet_gpu_model_runner.py`
- Official inference example: `inference/vllm_inference_example.py`
- SFT scripts: `script_examples/sft_stage1.sh`, `sft_stage2.sh`, `sft_stage3.sh`
- VLPO script: `RL/examples/vlpo_train.sh`

## Hardware policy
Development is intentionally split by cost.

### 4 x RTX 3090
Use for:
- baseline inference and environment validation;
- latent-state extraction/debugging;
- three-view (`I`, `I+`, `I-`) analysis;
- V0 sanity checks and small/medium SFT experiments;
- most implementation debugging.

The official SFT scripts are written for 8 GPUs with DeepSpeed ZeRO-2, so 4 x 24 GB 3090 is not treated as an exact reproduction of the author's training hardware. Any 3090 training recipe must be recorded separately and must not be silently presented as the official recipe.

### H200
Reserve for:
- full-scale training when 3090 memory/runtime becomes limiting;
- final multi-seed experiments;
- V2/VLPO-style RL and large rollout workloads.

## Baseline acceptance gates
Do not begin V0 until all checked items below are complete.

- [ ] Pinned Monet source is present locally and `git rev-parse HEAD` equals the pinned SHA.
- [ ] `Monet-7B` checkpoint is downloaded successfully.
- [ ] Official inference runs successfully on at least one example.
- [ ] The latent-mode switch is observed and the output format is understood.
- [ ] At least one selected benchmark/evaluation path is reproduced under documented settings.
- [ ] The verified baseline state is tagged/frozen in `V_COT`.

## Reproducibility rule
Every later experiment must record:
- V_COT commit SHA;
- Monet upstream SHA;
- model checkpoint;
- GPU type/count;
- environment/package versions;
- config and command;
- random seed;
- benchmark and metric.
