# Offline Server Workflow

> **Status: SUPERSEDED for the current 4 x RTX 3090 server.**
>
> The server was initially assumed to have no Internet access. The user later clarified that it can access ordinary mainland/domestic Internet resources and can install Python environments, while overseas services such as GitHub are unavailable or unreliable.
>
> Use `RESTRICTED_NETWORK_SETUP.md` for the current server. Keep this file only as a fallback for a truly air-gapped machine.

## Principle for a truly offline machine
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

```text
V_COT/
├── third_party/
│   └── Monet/
├── models/
│   └── Monet-7B/
├── offline/
│   ├── env/
│   └── wheelhouse/
├── scripts/
└── *.md
```

## Preferred environment strategy for a truly offline machine
Prepare the environment on another Linux x86_64 machine (or Linux VM/WSL2) and transfer it with `conda-pack`.

## Server-side verification after transfer

```bash
cd ~/work/V_COT

git -C third_party/Monet rev-parse HEAD
du -sh models/Monet-7B
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

The Monet SHA must be exactly:

```text
08939998d3d643a73a316e349faa34f420429153
```
