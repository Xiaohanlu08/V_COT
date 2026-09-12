#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
SITE_DIR="${ROOT_DIR}/scripts/monet_site"
SITE_FILE="${SITE_DIR}/sitecustomize.py"
VERIFY_PY="${ROOT_DIR}/scripts/06_force_latent_path.py"

if [[ ! -f "${SITE_FILE}" ]]; then
  echo "[ERROR] Missing spawn-safe Monet patch: ${SITE_FILE}" >&2
  echo "Create/transfer scripts/monet_site/sitecustomize.py before running Step 06." >&2
  exit 5
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VCOT_MONET_SITECUSTOMIZE=1
export LATENT_SIZE="${LATENT_SIZE:-10}"
export LATENT_START_ID=151666
export LATENT_END_ID=151667
export VLLM_USE_V1=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export MONET_MODEL_DIR="${MODEL_DIR}"

# sitecustomize.py must be on PYTHONPATH before Python starts so the Monet patch
# is applied in both the parent interpreter and spawned vLLM worker processes.
export PYTHONPATH="${SITE_DIR}:${MONET_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"

printf '\n[V_COT] Preflight: verifying sitecustomize patch before model loading...\n'
python - <<'PY'
import sitecustomize
import vllm.v1.worker.gpu_model_runner as g
print('sitecustomize_file =', sitecustomize.__file__)
print('gpu_model_runner_file =', g.__file__)
assert 'scripts/monet_site/sitecustomize.py' in sitecustomize.__file__, sitecustomize.__file__
assert 'third_party/Monet/inference/vllm/monet_gpu_model_runner.py' in g.__file__, g.__file__
print('[PREFLIGHT PASS] Monet custom runner is active in this interpreter.')
PY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  FREE_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (NR==1 || $2<min) {min=$2; idx=$1}} END{print idx}')"
  export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
fi

printf '\n[V_COT] Forced Monet latent-path diagnostic\n'
printf '[V_COT] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[V_COT] LATENT_SIZE=%s\n' "${LATENT_SIZE}"
printf '[V_COT] sitecustomize=%s\n\n' "${SITE_FILE}"

cd "${MONET_DIR}"
python "${VERIFY_PY}"
