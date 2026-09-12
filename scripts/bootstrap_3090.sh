#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
MONET_REPO="https://github.com/NOVAglow646/Monet.git"
MONET_PROXY_REPO="https://gh-proxy.com/https://github.com/NOVAglow646/Monet.git"
MONET_SHA="08939998d3d643a73a316e349faa34f420429153"
PIP_INDEX_URL_DEFAULT="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
HF_ENDPOINT_DEFAULT="https://hf-mirror.net"
CONDA_MAIN_DEFAULT="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"

export PIP_INDEX_URL="${PIP_INDEX_URL:-${PIP_INDEX_URL_DEFAULT}}"
export HF_ENDPOINT="${HF_ENDPOINT:-${HF_ENDPOINT_DEFAULT}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "${ROOT_DIR}/third_party" "${ROOT_DIR}/models" "${ROOT_DIR}/logs"

printf '\n[V_COT] root: %s\n' "${ROOT_DIR}"
printf '[V_COT] conda mirror: %s\n' "${CONDA_MAIN_DEFAULT}"
printf '[V_COT] pip mirror: %s\n' "${PIP_INDEX_URL}"
printf '[V_COT] HF mirror: %s\n\n' "${HF_ENDPOINT}"

if ! command -v conda >/dev/null 2>&1; then
  echo '[ERROR] conda was not found. Activate a shell with conda first.' >&2
  exit 1
fi

# Create a dedicated environment using a mainland mirror without modifying ~/.condarc.
if ! conda env list | awk '{print $1}' | grep -qx 'vcot'; then
  echo '[1/6] Creating conda env: vcot (Python 3.10) through TUNA mirror'
  conda create -y -n vcot python=3.10 --override-channels -c "${CONDA_MAIN_DEFAULT}"
else
  echo '[1/6] Conda env vcot already exists; keeping it.'
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

printf '\n[2/6] Installing basic Python tooling through mirror\n'
python -m pip install --upgrade pip -i "${PIP_INDEX_URL}"
python -m pip install -i "${PIP_INDEX_URL}" huggingface_hub

printf '\n[3/6] Preparing pinned Monet source\n'
if [[ ! -d "${MONET_DIR}/.git" ]]; then
  # Prefer the GitHub proxy. The proxy URL must wrap the full original GitHub URL.
  if git -c http.version=HTTP/1.1 clone --no-checkout "${MONET_PROXY_REPO}" "${MONET_DIR}"; then
    echo '[INFO] Monet cloned through gh-proxy.'
  else
    echo '[WARN] GitHub proxy failed; falling back to official GitHub over HTTP/1.1.'
    rm -rf "${MONET_DIR}"
    git -c http.version=HTTP/1.1 clone --no-checkout "${MONET_REPO}" "${MONET_DIR}"
  fi
fi

# Fetch is best-effort because the pinned commit may already be present after clone.
git -C "${MONET_DIR}" -c http.version=HTTP/1.1 fetch --all --tags --prune || true
git -C "${MONET_DIR}" checkout --detach "${MONET_SHA}"
ACTUAL_SHA="$(git -C "${MONET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${MONET_SHA}" ]]; then
  echo "[ERROR] Monet SHA mismatch: ${ACTUAL_SHA}" >&2
  exit 2
fi
echo "[OK] Monet pinned at ${ACTUAL_SHA}"

printf '\n[4/6] Installing Monet SFT/inference Python requirements\n'
python -m pip install -r "${MONET_DIR}/requirements.txt" -i "${PIP_INDEX_URL}"

printf '\n[5/6] Downloading Monet-7B through Hugging Face mirror\n'
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "[INFO] Existing model directory detected: ${MODEL_DIR}"
else
  if command -v hf >/dev/null 2>&1; then
    hf download NOVAglow646/Monet-7B --local-dir "${MODEL_DIR}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download NOVAglow646/Monet-7B --local-dir "${MODEL_DIR}"
  else
    echo '[ERROR] Neither hf nor huggingface-cli is available.' >&2
    exit 3
  fi
fi

printf '\n[6/6] Capturing environment snapshot\n'
{
  echo "date=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "monet_sha=${MONET_SHA}"
  echo "hf_endpoint=${HF_ENDPOINT}"
  echo "pip_index_url=${PIP_INDEX_URL}"
  echo
  echo '=== nvidia-smi ==='
  nvidia-smi || true
  echo
  echo '=== python ==='
  python --version
  echo
  echo '=== pip freeze ==='
  python -m pip freeze
} > "${ROOT_DIR}/logs/bootstrap_env.txt"

echo
printf '%s\n' '============================================================'
echo '[DONE] Bootstrap completed.'
echo "Monet source : ${MONET_DIR}"
echo "Monet model  : ${MODEL_DIR}"
echo "Env snapshot : ${ROOT_DIR}/logs/bootstrap_env.txt"
echo 'Next: run the verification commands documented in SETUP_3090.md.'
printf '%s\n' '============================================================'
