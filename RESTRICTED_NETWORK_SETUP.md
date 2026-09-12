# Restricted-Network Server Workflow

## Network facts
The target 4 x RTX 3090 server can access ordinary mainland/domestic Internet resources, including Python/Conda mirrors, but direct access to overseas services such as GitHub is unavailable or unreliable.

This means the server is **not fully offline**. Do not use an offline `conda-pack` workflow by default.

## Resource strategy
Use domestic-accessible mirrors for large dependencies and model weights, and transfer only small source repositories when necessary.

- Conda / Python packages: mainland mirrors such as TUNA.
- PyPI: `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`.
- Hugging Face checkpoints: `HF_ENDPOINT=https://hf-mirror.net` when reachable.
- Qwen checkpoints can also be obtained from ModelScope if required.
- GitHub source code: do not assume direct access from the server. Transfer V_COT and the pinned Monet source from an Internet-connected machine, or use a trusted domestic Git hosting mirror if one is established later.

## Required source trees
Server layout:

```text
~/work/V_COT/
├── third_party/
│   └── Monet/
├── models/
│   └── Monet-7B/
├── scripts/
└── *.md
```

Monet must remain pinned to:

```text
08939998d3d643a73a316e349faa34f420429153
```

## Environment creation on server
Once the V_COT and Monet source trees are present on the server, create the environment directly on the server through domestic mirrors.

```bash
conda create -y -n vcot python=3.10 \
  --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

python -m pip install --upgrade pip \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

python -m pip install -r ~/work/V_COT/third_party/Monet/requirements.txt \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

Do not change Monet package versions casually. The pinned requirements include `vllm==0.10.0`, `transformers==4.54.0`, and `trl==0.15.2`.

## Checkpoint download on server
Try the Hugging Face mirror first:

```bash
conda activate vcot
python -m pip install -U huggingface_hub \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

export HF_ENDPOINT=https://hf-mirror.net
hf download NOVAglow646/Monet-7B \
  --local-dir ~/work/V_COT/models/Monet-7B
```

If this endpoint is inaccessible from the server, transfer the checkpoint from another machine instead. Do not fall back to direct `huggingface.co` downloads without first confirming that overseas traffic is acceptable.

## Current synchronization model
Until a domestic Git mirror is configured:

```text
ChatGPT -> GitHub V_COT
                 ↓
        Internet-connected PC
                 ↓  SCP / SFTP / WinSCP
        restricted-network GPU server
```

Only source code needs this relay. Python packages and, preferably, model weights should be downloaded directly from domestic mirrors on the server.

## Verification gate
Before inference, verify:

```bash
conda activate vcot

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
PY

git -C ~/work/V_COT/third_party/Monet rev-parse HEAD
```

The Monet SHA must equal the pinned commit above.
