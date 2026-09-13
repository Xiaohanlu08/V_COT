#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
VLMEVAL_DIR="${ROOT_DIR}/third_party/VLMEvalKit"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
TENSOR_PATCHER="${ROOT_DIR}/scripts/12_make_tensor_capture_runner.py"
TRIGGER_PATCHER="${ROOT_DIR}/scripts/14_make_fixed_trigger_runner.py"
PY_SCRIPT="${ROOT_DIR}/scripts/14_vstar_fixed_prefix_evidence_replay.py"
BASELINE_JSONL="${ROOT_DIR}/results/natural_trigger/vstar_n191_seed20260913.jsonl"
EXP5_TENSOR="${ROOT_DIR}/results/latent_capture/vstar_pos0_latents.pt"
POS_IMAGE="${ROOT_DIR}/results/evidence_pilot/vstar_pos0/I_positive_zoom.png"
NEG_IMAGE="${ROOT_DIR}/results/evidence_pilot/vstar_pos0/I_negative_mask.png"
LOG_DIR="${ROOT_DIR}/logs"
RESULT_DIR="${ROOT_DIR}/results/evidence_replay/vstar_pos0"
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
  "${EXP5_TENSOR}" \
  "${POS_IMAGE}" \
  "${NEG_IMAGE}" \
  "${TENSOR_PATCHER}" \
  "${TRIGGER_PATCHER}" \
  "${PY_SCRIPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[ERROR] Missing required file: ${required}" >&2
    exit 2
  fi
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vcot_vstar_fixed_replay.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${WORK_DIR}/Monet_models" "${WORK_DIR}/latent_dump"
cp "${MONET_DIR}/inference/vllm/monet_gpu_model_runner.py" \
   "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"
touch "${WORK_DIR}/Monet_models/__init__.py"

python "${TENSOR_PATCHER}" "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"
python "${TRIGGER_PATCHER}" "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"

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

LOG_FILE="${LOG_DIR}/14_vstar_fixed_prefix_evidence_replay.log"

echo "[V_COT] VStarBench position-0 fixed-prefix/fixed-trigger evidence replay"
echo "[V_COT] Monet commit: ${ACTUAL_SHA}"
echo "[V_COT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[V_COT] LATENT_SIZE=${LATENT_SIZE}"
echo "[V_COT] FORCE_LATENT_START_ONCE=${VCOT_FORCE_LATENT_START_ONCE}"
echo "[V_COT] Dump dir=${VCOT_LATENT_DUMP_DIR}"
echo "[V_COT] Results=${RESULT_DIR}"
echo "[V_COT] Log=${LOG_FILE}"
echo

cd "${VLMEVAL_DIR}"
python "${PY_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
