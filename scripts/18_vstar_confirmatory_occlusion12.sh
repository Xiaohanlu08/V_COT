#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
VLMEVAL_DIR="${ROOT_DIR}/third_party/VLMEvalKit"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
TENSOR_PATCHER="${ROOT_DIR}/scripts/12_make_tensor_capture_runner.py"
CONFIRM_PATCHER="${ROOT_DIR}/scripts/18_make_confirmatory_runner.py"
PY_SCRIPT="${ROOT_DIR}/scripts/18_vstar_confirmatory_occlusion12.py"
BASELINE_JSONL="${ROOT_DIR}/results/natural_trigger/vstar_n191_seed20260913.jsonl"
COHORT_JSON="${ROOT_DIR}/results/replication_cohort/vstar_direct_attributes_confirmatory12.json"
PROTOCOL="${ROOT_DIR}/protocols/CONFIRMATORY_OCCLUSION_12.md"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/confirmatory_occlusion12"
EXPECTED_MONET_SHA="08939998d3d643a73a316e349faa34f420429153"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

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
  "${BASELINE_JSONL}" \
  "${COHORT_JSON}" \
  "${PROTOCOL}" \
  "${TENSOR_PATCHER}" \
  "${CONFIRM_PATCHER}" \
  "${PY_SCRIPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] Missing required file: ${required}" >&2
    exit 2
  fi
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vcot_confirmatory12.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${WORK_DIR}/Monet_models" "${WORK_DIR}/latent_dump" "${WORK_DIR}/preforce_dump"
cp "${MONET_DIR}/inference/vllm/monet_gpu_model_runner.py" \
   "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"
touch "${WORK_DIR}/Monet_models/__init__.py"

python "${TENSOR_PATCHER}" "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"
python "${CONFIRM_PATCHER}" "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"

cat > "${WORK_DIR}/sitecustomize.py" <<'PYCODE'
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
export VCOT_LATENT_TENSOR_DUMP=1
export VCOT_LATENT_DUMP_DIR="${WORK_DIR}/latent_dump"
export VCOT_PREFORCE_DUMP_DIR="${WORK_DIR}/preforce_dump"
export VCOT_FORCE_LATENT_START_ONCE=1

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

LOG_FILE="${LOG_DIR}/18_vstar_confirmatory_occlusion12.log"

echo "[V_COT] Confirmatory 12-sample full-image target-vs-sham occlusion test"
echo "[V_COT] Frozen cohort SHA=f65ebdbefc8a73276877f4096d5cb8897339447c91a31dc68322a811dda487e4"
echo "[V_COT] Monet commit=${ACTUAL_SHA}"
echo "[V_COT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[V_COT] 32 sham masks/sample; seed rule=20260913+position"
echo "[V_COT] Primary test is pre-specified in protocols/CONFIRMATORY_OCCLUSION_12.md"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${VLMEVAL_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
