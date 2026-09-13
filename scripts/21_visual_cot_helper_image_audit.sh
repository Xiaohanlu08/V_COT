#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT_DIR}/scripts/21_visual_cot_helper_image_audit.py"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/v0_prep/visual_cot_helper_audit"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Missing ${PY_SCRIPT}" >&2
  exit 1
fi

LOG_FILE="${LOG_DIR}/21_visual_cot_helper_image_audit.log"

echo "[V_COT] Visual_CoT helper-image visual audit"
echo "[V_COT] Deterministic 12-sample audit, seed=20260913"
echo "[V_COT] Downloads only 24 small image files; no model weights, no GPU inference"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${ROOT_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
