#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_MAIN="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
PIP_INDEX="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
HF_ENDPOINT_URL="https://hf-mirror.net"

mkdir -p "${ROOT_DIR}/logs"

printf '\n[V_COT] restricted-network environment preparation\n'
printf '[V_COT] root: %s\n' "${ROOT_DIR}"
printf '[V_COT] conda mirror: %s\n' "${CONDA_MAIN}"
printf '[V_COT] pip mirror: %s\n' "${PIP_INDEX}"
printf '[V_COT] HF mirror: %s\n\n' "${HF_ENDPOINT_URL}"

if ! command -v conda >/dev/null 2>&1; then
  echo '[ERROR] conda was not found in PATH.' >&2
  exit 1
fi

printf '%s\n' '================ SERVER GPU ================'
nvidia-smi || true

printf '\n%s\n' '================ CONDA ================'
conda --version
conda env list

if ! conda env list | awk '{print $1}' | grep -qx 'vcot'; then
  echo
  echo '[1/3] Creating vcot (Python 3.10) through TUNA...'
  conda create -y -n vcot python=3.10 \
    --override-channels \
    -c "${CONDA_MAIN}"
else
  echo
  echo '[1/3] vcot already exists; keeping existing environment.'
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

printf '\n[2/3] Installing only lightweight bootstrap tools through TUNA PyPI...\n'
python -m pip install --upgrade pip -i "${PIP_INDEX}"
python -m pip install -U huggingface_hub -i "${PIP_INDEX}"

printf '\n[3/3] Capturing compatibility probe...\n'
{
  echo "date=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo
  echo '=== GPU SUMMARY ==='
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  echo
  echo '=== NVIDIA-SMI ==='
  nvidia-smi || true
  echo
  echo '=== NVCC ==='
  nvcc --version || true
  echo
  echo '=== PYTHON ==='
  python --version
  echo
  echo '=== PIP ==='
  python -m pip --version
  echo
  echo '=== TORCH BEFORE MONET INSTALL ==='
  python - <<'PY'
try:
    import torch
    print('torch:', torch.__version__)
    print('torch cuda:', torch.version.cuda)
    print('cuda available:', torch.cuda.is_available())
    print('gpu count:', torch.cuda.device_count())
except Exception as e:
    print('torch not installed or import failed:', repr(e))
PY
  echo
  echo '=== MIRROR CHECK ==='
  curl -I --max-time 10 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/pip/ || true
  echo
  curl -I --max-time 10 https://hf-mirror.net || true
} | tee "${ROOT_DIR}/logs/01_prepare_restricted_env.log"

cat <<'EOF'

============================================================
[DONE] Lightweight environment preparation completed.

IMPORTANT:
- Monet heavy requirements have NOT been installed yet.
- Monet-7B has NOT been downloaded yet.
- This is intentional: first inspect GPU driver/CUDA compatibility.

Send back:
  ~/work/V_COT/logs/01_prepare_restricted_env.log
or paste the complete terminal output.
============================================================
EOF
