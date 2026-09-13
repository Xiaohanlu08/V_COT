#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT_DIR}/scripts/20_monet_sft125k_schema_audit.py"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/v0_prep"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${ROOT_DIR}/data"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.net}"
export HF_HUB_DISABLE_TELEMETRY=1

LOG_FILE="${LOG_DIR}/20_monet_sft125k_schema_audit.log"

echo "[V_COT] Monet-SFT-125K metadata-only schema audit"
echo "[V_COT] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[V_COT] Downloads only six train.json files + small Stage-3 metadata"
echo "[V_COT] No image archives, no model weight shards, no GPU inference"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${ROOT_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
