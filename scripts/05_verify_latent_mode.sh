#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
VERIFY_PY="${ROOT_DIR}/scripts/05_verify_latent_mode.py"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  FREE_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (NR==1 || $2<min) {min=$2; idx=$1}} END{print idx}')"
  export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
fi

export LATENT_SIZE="${LATENT_SIZE:-10}"
export LATENT_START_ID=151666
export LATENT_END_ID=151667
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MONET_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
export MONET_MODEL_DIR="${MODEL_DIR}"

printf '\n[V_COT] Monet latent-mode diagnostic\n'
printf '[V_COT] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[V_COT] LATENT_SIZE=%s\n\n' "${LATENT_SIZE}"

cd "${MONET_DIR}"
python "${VERIFY_PY}"
