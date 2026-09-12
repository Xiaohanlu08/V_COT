#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
VLMEVAL_DIR="${ROOT_DIR}/third_party/VLMEvalKit"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
PY_SCRIPT="${ROOT_DIR}/scripts/09_vstar_natural_trigger_scan.py"
RESULT_DIR="${ROOT_DIR}/results/natural_trigger"
LOG_DIR="${ROOT_DIR}/logs"
EXPECTED_MONET_SHA="08939998d3d643a73a316e349faa34f420429153"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

ACTUAL_SHA="$(git -C "${MONET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_MONET_SHA}" ]]; then
  echo "[ERROR] Monet commit mismatch." >&2
  echo "expected: ${EXPECTED_MONET_SHA}" >&2
  echo "actual  : ${ACTUAL_SHA}" >&2
  exit 1
fi

for required in \
  "${MODEL_DIR}/config.json" \
  "${VLMEVAL_DIR}/vlmeval/vlm/qwen2_vl/model.py" \
  "${PY_SCRIPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] Missing required file: ${required}" >&2
    exit 2
  fi
done

VCOT_N="${VCOT_N:-20}"
VCOT_SEED="${VCOT_SEED:-20260913}"
LOG_FILE="${LOG_DIR}/09_vstar_natural_trigger_n${VCOT_N}_seed${VCOT_SEED}.log"
OUT_JSONL="${RESULT_DIR}/vstar_n${VCOT_N}_seed${VCOT_SEED}.jsonl"
OUT_SUMMARY="${RESULT_DIR}/vstar_n${VCOT_N}_seed${VCOT_SEED}_summary.json"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vcot_vstar_scan.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${WORK_DIR}/Monet_models"
cp "${MONET_DIR}/inference/vllm/monet_gpu_model_runner.py" \
   "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"
touch "${WORK_DIR}/Monet_models/__init__.py"

cat > "${WORK_DIR}/sitecustomize.py" <<'PYCODE'
# sitecustomize.py (top-level)
# Runs in every Python process (parent + spawned workers)

import os, sys, importlib
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
os.environ["LATENT_START_ID"] = "151666"
os.environ["LATENT_END_ID"] = "151667"
sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module("Monet_models.monet_gpu_model_runner")
PYCODE

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export VLLM_NO_USAGE_STATS=1
export LATENT_START_ID=151666
export LATENT_END_ID=151667
export LATENT_SIZE=10
export TOKENIZERS_PARALLELISM=false
export VCOT_ROOT="${ROOT_DIR}"
export VCOT_N
export VCOT_SEED
export VCOT_OUT_JSONL="${OUT_JSONL}"
export VCOT_OUT_SUMMARY="${OUT_SUMMARY}"

# Optional explicit override, e.g.:
#   VCOT_GPUS=4,5,6,7 VCOT_N=20 bash scripts/09_vstar_natural_trigger_scan.sh
if [[ -n "${VCOT_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${VCOT_GPUS}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
      awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $2, $1}' | \
      sort -n -k1,1 | head -n 4 | awk '{print $2}' | paste -sd, -
  )"
  export CUDA_VISIBLE_DEVICES
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] Could not select GPUs." >&2
  exit 3
fi

export PYTHONPATH="${WORK_DIR}:${VLMEVAL_DIR}:${PYTHONPATH:-}"

echo "[V_COT] VStarBench natural latent-trigger scan"
echo "[V_COT] Monet commit: ${ACTUAL_SHA}"
echo "[V_COT] VLMEvalKit: ${VLMEVAL_DIR}"
echo "[V_COT] Model: ${MODEL_DIR}"
echo "[V_COT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[V_COT] LATENT_SIZE=${LATENT_SIZE}"
echo "[V_COT] VCOT_N=${VCOT_N}"
echo "[V_COT] VCOT_SEED=${VCOT_SEED}"
echo "[V_COT] JSONL=${OUT_JSONL}"
echo "[V_COT] SUMMARY=${OUT_SUMMARY}"
echo "[V_COT] LOG=${LOG_FILE}"
echo

cd "${VLMEVAL_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
