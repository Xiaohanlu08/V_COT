#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
HF_ENDPOINT_URL="https://hf-mirror.net"
HF_HOME_DIR="${ROOT_DIR}/.cache/huggingface"

mkdir -p "${ROOT_DIR}/models" "${ROOT_DIR}/logs" "${HF_HOME_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo '[ERROR] conda was not found in PATH.' >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

export HF_ENDPOINT="${HF_ENDPOINT_URL}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_DISABLE_XET=1

printf '\n[V_COT] Downloading Monet-7B\n'
printf '[V_COT] HF endpoint: %s\n' "${HF_ENDPOINT}"
printf '[V_COT] local model dir: %s\n\n' "${MODEL_DIR}"

python - <<'PY'
import torch, transformers, trl, vllm
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('trl:', trl.__version__)
print('vllm:', vllm.__version__)
assert torch.__version__.split('+')[0] == '2.7.1'
assert transformers.__version__ == '4.54.0'
assert trl.__version__ == '0.15.2'
assert vllm.__version__ == '0.10.0'
PY

if ! command -v hf >/dev/null 2>&1; then
  echo '[ERROR] hf CLI not found in vcot environment.' >&2
  exit 2
fi

echo '[1/2] Downloading/resuming NOVAglow646/Monet-7B through hf-mirror...'
hf download NOVAglow646/Monet-7B \
  --repo-type model \
  --local-dir "${MODEL_DIR}"

echo
echo '[2/2] Verifying local checkpoint layout...'
python - <<PY
from pathlib import Path
import json

model_dir = Path(r"${MODEL_DIR}")
required = [
    model_dir / 'config.json',
    model_dir / 'generation_config.json',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit('Missing required files: ' + ', '.join(missing))

weights = sorted(model_dir.glob('*.safetensors'))
if not weights:
    raise SystemExit('No .safetensors model weights found.')

with open(model_dir / 'config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

print('model_type:', cfg.get('model_type'))
print('architectures:', cfg.get('architectures'))
print('safetensors files:', len(weights))
print('total checkpoint size GiB:', round(sum(p.stat().st_size for p in weights) / 1024**3, 2))
print('model directory:', model_dir)
PY

du -sh "${MODEL_DIR}" | tee "${ROOT_DIR}/logs/03_monet7b_size.txt"
find "${MODEL_DIR}" -maxdepth 1 -type f -printf '%f\n' | sort | head -100

echo
printf '%s\n' '============================================================'
echo '[DONE] Monet-7B download and structural verification completed.'
echo 'Next: run scripts/04_monet_smoke_inference.sh.'
printf '%s\n' '============================================================'
