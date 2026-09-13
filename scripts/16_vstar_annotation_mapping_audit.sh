#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVAL_DIR="${ROOT_DIR}/third_party/VLMEvalKit"
PY_SCRIPT="${ROOT_DIR}/scripts/16_vstar_annotation_mapping_audit.py"
BASELINE_JSONL="${ROOT_DIR}/results/natural_trigger/vstar_n191_seed20260913.jsonl"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/annotation_audit"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

for required in \
  "${PY_SCRIPT}" \
  "${BASELINE_JSONL}" \
  "${VLMEVAL_DIR}/vlmeval/dataset/image_mcq.py"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] Missing required file: ${required}" >&2
    exit 1
  fi
done

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.net}"
export PYTHONPATH="${VLMEVAL_DIR}:${PYTHONPATH:-}"

LOG_FILE="${LOG_DIR}/16_vstar_annotation_mapping_audit.log"

echo "[V_COT] VStarBench triggered-72 annotation mapping audit"
echo "[V_COT] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[V_COT] No model loading / no GPU inference"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${ROOT_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
