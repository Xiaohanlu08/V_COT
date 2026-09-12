# 4 x RTX 3090 Baseline Setup

This document is for the first project gate only: reproduce Monet inference and inspect the latent reasoning path. It is intentionally not the final training recipe.

## 1. Server directory
The expected working directory is:

```bash
~/work/V_COT
```

If this directory is empty, clone this repository into the current directory:

```bash
cd ~/work/V_COT
git clone https://github.com/Xiaohanlu08/V_COT.git .
```

The repository itself is tiny, so direct GitHub access is acceptable here. Large external assets use mirrors in later steps.

## 2. Optional: configure Conda mirror
If your Conda is not already configured to a local/mainland mirror, back up `~/.condarc` first and then use the TUNA mirror configuration below.

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

Then refresh the index:

```bash
conda clean -i
```

Do not overwrite an existing working Conda configuration without backing it up.

## 3. Bootstrap
Run:

```bash
cd ~/work/V_COT
chmod +x scripts/bootstrap_3090.sh
bash scripts/bootstrap_3090.sh 2>&1 | tee logs/bootstrap_console.log
```

The script does four important things:

1. creates a `vcot` Python 3.10 Conda environment if needed;
2. installs Python packages through the TUNA PyPI mirror;
3. clones Monet, preferring `gh-proxy.com` and falling back to official GitHub, then checks out the exact pinned SHA;
4. downloads `NOVAglow646/Monet-7B` through `HF_ENDPOINT=https://hf-mirror.net`.

The SHA check is mandatory. A mirror is used only as a transport path; the baseline source identity is still the official Monet commit recorded in `BASELINE.md`.

## 4. Expected directories after bootstrap

```text
V_COT/
├── third_party/
│   └── Monet/          # ignored by V_COT Git; pinned upstream clone
├── models/
│   └── Monet-7B/       # ignored by V_COT Git
├── logs/
│   └── bootstrap_env.txt
├── scripts/
└── *.md
```

Model checkpoints and external source trees are not committed into V_COT.

## 5. Verify hardware and pinned source

```bash
conda activate vcot
cd ~/work/V_COT

nvidia-smi

git -C third_party/Monet rev-parse HEAD
```

The second command must print exactly:

```text
08939998d3d643a73a316e349faa34f420429153
```

Verify the model directory:

```bash
ls -lh models/Monet-7B | head -30
```

## 6. Do not start training yet
The first successful checkpoint is not a training checkpoint. It is:

- environment works;
- model files are complete;
- upstream SHA matches;
- official inference can run;
- latent mode can be observed.

Only after those checks will we write the 3090-specific inference command and, later, V0 training configuration.

## 7. What to send back after bootstrap
Send the terminal output of these commands:

```bash
cd ~/work/V_COT
conda activate vcot

echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

echo '=== MONET SHA ==='
git -C third_party/Monet rev-parse HEAD

echo '=== PYTHON ==='
python --version

echo '=== TORCH ==='
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 1024**3, 2), 'GiB')
PY

echo '=== MODEL FILES ==='
du -sh models/Monet-7B
ls models/Monet-7B | head -30
```

Do not make additional package/version changes after an error unless we first inspect the error, because Monet depends on modified Transformers/vLLM code and casual upgrades can create hidden incompatibilities.
