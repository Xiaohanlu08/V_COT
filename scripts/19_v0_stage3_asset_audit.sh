#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT_DIR}/scripts/19_v0_stage3_asset_audit.py"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/v0_prep"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Missing required file: ${PY_SCRIPT}" >&2
  exit 1
fi

LOG_FILE="${LOG_DIR}/19_v0_stage3_asset_audit.log"

echo "[V_COT] V0 Stage-3 source / training asset audit"
echo "[V_COT] No model loading, no GPU inference, no package installation"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${ROOT_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
