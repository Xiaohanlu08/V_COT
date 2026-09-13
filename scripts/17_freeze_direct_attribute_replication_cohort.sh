#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT_DIR}/scripts/17_freeze_direct_attribute_replication_cohort.py"
MAPPING_JSONL="${ROOT_DIR}/results/annotation_audit/vstar_triggered72_annotation_mapping.jsonl"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/replication_cohort"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

for required in "${PY_SCRIPT}" "${MAPPING_JSONL}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] Missing required file: ${required}" >&2
    exit 1
  fi
done

LOG_FILE="${LOG_DIR}/17_freeze_direct_attribute_replication_cohort.log"

echo "[V_COT] Freeze outcome-blind VStarBench direct-attribute replication cohort"
echo "[V_COT] No model loading / no GPU inference"
echo "[V_COT] Development sample position 0 is excluded"
echo "[V_COT] Selection seed=20260913, cohort size=12"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${ROOT_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
