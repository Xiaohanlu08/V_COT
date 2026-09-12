#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"

mkdir -p "${ROOT_DIR}/logs"

if ! command -v conda >/dev/null 2>&1; then
  echo '[ERROR] conda was not found in PATH.' >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "[ERROR] Monet-7B checkpoint is missing at ${MODEL_DIR}." >&2
  echo 'Run scripts/03_download_monet7b.sh first.' >&2
  exit 2
fi

if [[ ! -f "${MONET_DIR}/images/example_question.png" ]]; then
  echo '[ERROR] Official Monet example image is missing.' >&2
  exit 3
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

# Use one known-free physical 3090 for the first smoke test.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LATENT_SIZE="${LATENT_SIZE:-10}"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MONET_DIR}:${PYTHONPATH:-}"
export MONET_MODEL_DIR="${MODEL_DIR}"

printf '\n[V_COT] Monet smoke inference\n'
printf '[V_COT] CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '[V_COT] LATENT_SIZE=%s\n' "${LATENT_SIZE}"
printf '[V_COT] model=%s\n\n' "${MODEL_DIR}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

cd "${MONET_DIR}"

python - <<'PY'
import os
import re
from pathlib import Path

# Must be imported before vLLM so Monet can patch the runner.
import inference.apply_vllm_monet
import PIL.Image
from transformers import AutoProcessor
from inference.load_and_gen_vllm import (
    vllm_mllm_init,
    vllm_mllm_process_batch_from_messages,
    vllm_generate,
)

model_path = os.environ['MONET_MODEL_DIR']
image_path = Path('images/example_question.png')


def clean_latent_text(s: str) -> str:
    pattern = re.compile(r'(<abs_vis_token>)(.*?)(</abs_vis_token>)', flags=re.DOTALL)
    return pattern.sub(r'\1<latent>\3', s)

print('[SMOKE] initializing Monet/vLLM...')
mllm, sampling_params = vllm_mllm_init(
    model_path,
    tp=1,
    gpu_memory_utilization=0.80,
    max_model_len=4096,
)

print('[SMOKE] loading processor...')
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

conversations = [[{
    'role': 'user',
    'content': [
        {
            'type': 'text',
            'text': (
                'Question: Which car has the longest rental period? The choices are listed below:\n'
                '(A) DB11 COUPE.\n'
                '(B) V12 VANTAGES COUPES.\n'
                '(C) VANQUISH VOLANTE.\n'
                '(D) V12 VOLANTE.\n'
                '(E) The image does not feature the time. '
                'Put your final answer in \\boxed{}.'
            ),
        },
        {
            'type': 'image',
            'image': PIL.Image.open(image_path).convert('RGB'),
        },
    ],
}]]

print('[SMOKE] preparing multimodal input...')
inputs = vllm_mllm_process_batch_from_messages(conversations, processor)

print('[SMOKE] generating...')
output = vllm_generate(inputs, sampling_params, mllm)
raw = output[0].outputs[0].text
cleaned = clean_latent_text(raw)

print('\n================ RAW OUTPUT ================')
print(raw)
print('\n============== CLEANED OUTPUT ==============')
print(cleaned)
print('\n=============== LATENT CHECK ===============')
print('contains <abs_vis_token>:', '<abs_vis_token>' in raw)
print('contains </abs_vis_token>:', '</abs_vis_token>' in raw)
print('LATENT_SIZE:', os.environ.get('LATENT_SIZE'))

assert len(raw.strip()) > 0, 'Model returned an empty output.'
print('\n[SMOKE PASS] Monet produced a non-empty output.')
PY

echo
printf '%s\n' '============================================================'
echo '[DONE] Monet official-example smoke inference completed.'
echo 'Inspect RAW OUTPUT and LATENT CHECK before moving to benchmarks.'
printf '%s\n' '============================================================'
