# Offline Server Workflow

The target GPU server has no Internet access. Therefore the server must not depend on GitHub, PyPI, Conda mirrors, Hugging Face, or ModelScope at runtime.

## Principle
Use an Internet-connected staging machine to download and verify all external assets, then transfer them to the server over the user's available local/SSH/SFTP path.

The server-side working directory remains:

```text
~/work/V_COT
```

## Required offline assets

1. This repository (`Xiaohanlu08/V_COT`).
2. Monet source pinned to commit:
   `08939998d3d643a73a316e349faa34f420429153`.
3. Monet-7B checkpoint (`NOVAglow646/Monet-7B`).
4. A Linux-compatible Python environment or wheelhouse for Monet dependencies.

The official Monet SFT/inference requirements at the pinned commit include `vllm==0.10.0`, `transformers==4.54.0`, `trl==0.15.2`, PyTorch, torchvision, DeepSpeed, Ray, qwen-vl-utils, and related packages. Do not install arbitrary newer versions.

## Recommended transfer layout
Prepare the following layout on the Internet-connected staging machine before upload:

```text
V_COT/
├── third_party/
│   └── Monet/
├── models/
│   └── Monet-7B/
├── offline/
│   ├── env/              # optional packed Conda environment
│   └── wheelhouse/       # optional Linux wheels
├── scripts/
└── *.md
```

## Preferred environment strategy
The most robust strategy is to prepare the environment on another Linux x86_64 machine (or Linux VM/WSL2) and transfer it with `conda-pack`.

Why: downloading wheels on Windows is not sufficient for Linux-only binary packages such as PyTorch/vLLM/DeepSpeed dependencies.

If no Linux staging machine is available, first transfer source + model to the server and inspect the server's existing CUDA/PyTorch stack before deciding how to build the offline environment.

## Server-side verification after transfer

```bash
cd ~/work/V_COT

echo '=== V_COT ==='
ls -lah

echo '=== MONET SHA ==='
git -C third_party/Monet rev-parse HEAD

echo '=== MODEL ==='
du -sh models/Monet-7B
ls models/Monet-7B | head -30

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

The Monet SHA must be exactly:

```text
08939998d3d643a73a316e349faa34f420429153
```

## Version-control workflow with an offline server
GitHub remains the project authority, but the GPU server is an execution replica.

Normal cycle:

```text
ChatGPT/GitHub -> local staging machine -> offline GPU server
                                      <- results/logs/checkpoints
```

Do not make the offline server the only copy of code changes. Any verified source change should eventually be committed to GitHub through the Internet-connected machine.
