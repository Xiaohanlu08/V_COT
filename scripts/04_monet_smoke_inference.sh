#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
SMOKE_PY="${ROOT_DIR}/scripts/04_monet_smoke_inference.py"

mkdir -p "${ROOT_DIR}/logs"

if ! command -v conda >/dev/null 2>&1; then
  echo '[ERROR] conda was not found in PATH.' >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] Monet-7B checkpoint is missing at ${MODEL_DIR}." >&2
  echo 'Run scripts/03_download_monet7b.sh first.' >&2
  exit 2
fi

if [[ ! -f "${MONET_DIR}/images/example_question.png" ]]; then
  echo '[ERROR] Official Monet example image is missing.' >&2
  exit 3
fi

if [[ ! -f "${SMOKE_PY}" ]]; then
  echo "[ERROR] Smoke Python entrypoint missing: ${SMOKE_PY}" >&2
  exit 4
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

# vLLM 0.10.0 uses spawn when CUDA has been initialized. A real Python file with
# an if __name__ == '__main__' guard is therefore required; stdin execution is invalid.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Pick the currently least-used physical GPU unless the caller explicitly chooses one.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  FREE_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (NR==1 || $2<min) {min=$2; idx=$1}} END{print idx}')"
  export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
fi

export LATENT_SIZE="${LATENT_SIZE:-10}"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MONET_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
export MONET_MODEL_DIR="${MODEL_DIR}"

printf '\n[V_COT] Monet smoke inference\n'
printf '[V_COT] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[V_COT] LATENT_SIZE=%s\n' "${LATENT_SIZE}"
printf '[V_COT] model=%s\n\n' "${MODEL_DIR}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

cd "${MONET_DIR}"
python "${SMOKE_PY}"

echo
printf '%s\n' '============================================================'
echo '[DONE] Monet official-example smoke inference completed.'
echo 'Inspect RAW OUTPUT and LATENT CHECK before moving to benchmarks.'
printf '%s\n' '============================================================'
