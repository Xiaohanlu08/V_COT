#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_SCRIPT="${ROOT_DIR}/scripts/20b_monet_sft125k_schema_audit_direct_get.py"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/v0_prep"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${ROOT_DIR}/data"

LOG_FILE="${LOG_DIR}/20b_monet_sft125k_schema_audit_direct_get.log"

echo "[V_COT] Monet-SFT-125K schema audit — direct GET fallback"
echo "[V_COT] Bypasses huggingface_hub HEAD metadata requests"
echo "[V_COT] Endpoint order: hf-mirror.com -> hf-mirror.net -> huggingface.co"
echo "[V_COT] No package installation, no image archives, no model weight shards, no GPU inference"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${ROOT_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
