#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
EXPECTED_MONET_SHA="08939998d3d643a73a316e349faa34f420429153"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

ACTUAL_SHA="$(git -C "${MONET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_MONET_SHA}" ]]; then
  echo "[ERROR] Monet commit mismatch." >&2
  echo "expected: ${EXPECTED_MONET_SHA}" >&2
  echo "actual  : ${ACTUAL_SHA}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] Monet-7B checkpoint missing: ${MODEL_DIR}" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vcot_monet_latent_path.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${WORK_DIR}/Monet_models"
cp "${MONET_DIR}/inference/vllm/monet_gpu_model_runner.py" \
   "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"

# Monet README patch body. The README command says `sitecustomized.py`, while
# its own comment says `sitecustomize.py`; only that filename typo is corrected.
cat > "${WORK_DIR}/sitecustomize.py" <<'PYCODE'
# sitecustomize.py (top-level)
# Runs in every Python process (parent + spawned workers)

import os, sys, importlib
os.environ["VLLM_USE_V1"] = "1"  # force V1 engine if desired
os.environ["VLLM_NO_USAGE_STATS"] = "1"  # disable usage stats
workspace = os.path.abspath(".")
old_path = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{old_path}" if old_path else workspace
os.environ["LATENT_START_ID"] = "151666"
os.environ["LATENT_END_ID"] = "151667"
sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module("Monet_models.monet_gpu_model_runner")
PYCODE

cat > "${WORK_DIR}/run_latent_path.py" <<'PYCODE'
import os
from pathlib import Path


def main():
    import PIL.Image
    import vllm.v1.worker.gpu_model_runner as runner
    from transformers import AutoProcessor
    from inference.load_and_gen_vllm import (
        vllm_generate,
        vllm_mllm_init,
        vllm_mllm_process_batch_from_messages,
    )

    model_path = os.environ["MONET_MODEL_DIR"]
    image_path = Path(os.environ["MONET_DIR"]) / "images" / "example_question.png"

    print("=============== RUNNER CHECK ===============")
    print("runner_file:", Path(runner.__file__).resolve())
    print("runner_class_module:", runner.GPUModelRunner.__module__)
    print("LATENT_START_ID:", os.environ.get("LATENT_START_ID"))
    print("LATENT_END_ID:", os.environ.get("LATENT_END_ID"))
    print("LATENT_SIZE:", os.environ.get("LATENT_SIZE"))

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    start_id = processor.tokenizer.convert_tokens_to_ids("<abs_vis_token>")
    end_id = processor.tokenizer.convert_tokens_to_ids("</abs_vis_token>")
    assert start_id == 151666, start_id
    assert end_id == 151667, end_id

    print("\n[TEST] initializing Monet/vLLM with official-style runner patch...")
    mllm, sampling_params = vllm_mllm_init(
        model_path,
        tp=1,
        gpu_memory_utilization=0.80,
        max_model_len=4096,
    )

    # Engineering-only deterministic test:
    # - sampler is allowed to choose only LATENT_START_ID;
    # - LATENT_SIZE=2;
    # - Monet runner should rewrite the third sampled start token to
    #   LATENT_END_ID when its latent state reaches the configured length.
    # A plain vLLM runner would return [start, start, start].
    sampling_params.allowed_token_ids = [start_id]
    sampling_params.temperature = 0.0
    sampling_params.max_tokens = 3

    conversations = [[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Engineering-only latent-path test. Inspect the image. "
                    "The generated token IDs are constrained externally; "
                    "this is not a benchmark prompt."
                ),
            },
            {
                "type": "image",
                "image": PIL.Image.open(image_path).convert("RGB"),
            },
        ],
    }]]

    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)
    output = vllm_generate(inputs, sampling_params, mllm)
    candidate = output[0].outputs[0]
    token_ids = list(candidate.token_ids)

    print("\n================ RAW OUTPUT ================")
    print(candidate.text)
    print("\n=============== TOKEN IDS ==================")
    print(token_ids)
    print("\n============== EXPECTATION =================")
    expected = [start_id, start_id, end_id]
    print("expected:", expected)
    print("observed:", token_ids)
    print("OFFICIAL_LATENT_PATH_PASS=", token_ids == expected)

    if token_ids != expected:
        raise SystemExit(
            f"Expected Monet latent state-machine output {expected}, got {token_ids}"
        )


if __name__ == "__main__":
    main()
PYCODE

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export VLLM_NO_USAGE_STATS=1
export LATENT_START_ID=151666
export LATENT_END_ID=151667
export LATENT_SIZE=2
export TOKENIZERS_PARALLELISM=false
export MONET_MODEL_DIR="${MODEL_DIR}"
export MONET_DIR="${MONET_DIR}"
export PYTHONPATH="${WORK_DIR}:${MONET_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  FREE_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (NR==1 || $2<min) {min=$2; idx=$1}} END{print idx}')"
  export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
fi

echo
echo "[V_COT] Official-style forced Monet latent-path test"
echo "[V_COT] Monet commit: ${ACTUAL_SHA}"
echo "[V_COT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[V_COT] LATENT_SIZE=${LATENT_SIZE}"
echo "[V_COT] temporary patch dir: ${WORK_DIR}"
echo

cd "${WORK_DIR}"
python run_latent_path.py
