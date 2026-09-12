#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/Monet-7B"
HF_ENDPOINT_URL="https://hf-mirror.net"
REPO_ID="NOVAglow646/Monet-7B"
TMP_LIST="${ROOT_DIR}/logs/03_monet7b_files.txt"

mkdir -p "${MODEL_DIR}" "${ROOT_DIR}/logs"

if ! command -v conda >/dev/null 2>&1; then
  echo '[ERROR] conda was not found in PATH.' >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

printf '\n[V_COT] Downloading Monet-7B with direct GET requests\n'
printf '[V_COT] HF endpoint: %s\n' "${HF_ENDPOINT_URL}"
printf '[V_COT] repository: %s\n' "${REPO_ID}"
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

# huggingface_hub performs HEAD metadata requests before GET. Some Cloudflare-backed
# mirrors omit Content-Length on HEAD responses, which makes huggingface_hub abort
# even though ordinary GET downloads work. Therefore this script intentionally
# bypasses `hf download` and downloads the public repo files with curl GET requests.

echo '[1/3] Resolving repository file list from mirror API...'
API_JSON="$(mktemp)"
trap 'rm -f "${API_JSON}"' EXIT

if curl -fsSL --retry 5 --retry-delay 2 --connect-timeout 15 \
  "${HF_ENDPOINT_URL}/api/models/${REPO_ID}" -o "${API_JSON}"; then
  python - "${API_JSON}" > "${TMP_LIST}" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
files = [x.get('rfilename') for x in data.get('siblings', []) if x.get('rfilename')]
for name in files:
    print(name)
PY
else
  echo '[WARN] Mirror API listing failed; using the known Monet-7B file manifest.'
  cat > "${TMP_LIST}" <<'EOF'
.gitattributes
README.md
added_tokens.json
chat_template.json
config.json
generation_config.json
merges.txt
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
preprocessor_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.json
EOF
fi

if [[ ! -s "${TMP_LIST}" ]]; then
  echo '[ERROR] Repository file list is empty.' >&2
  exit 2
fi

echo '[INFO] Files to ensure locally:'
cat "${TMP_LIST}"

urlencode_path() {
  python - "$1" <<'PY'
import sys, urllib.parse
print('/'.join(urllib.parse.quote(part, safe='') for part in sys.argv[1].split('/')))
PY
}

download_one() {
  local rel="$1"
  local dest="${MODEL_DIR}/${rel}"
  local part="${dest}.part"
  local encoded
  encoded="$(urlencode_path "${rel}")"
  local url="${HF_ENDPOINT_URL}/${REPO_ID}/resolve/main/${encoded}?download=true"

  mkdir -p "$(dirname "${dest}")"

  if [[ -s "${dest}" ]]; then
    echo "[SKIP] ${rel} already exists ($(du -h "${dest}" | cut -f1))."
    return 0
  fi

  echo
  echo "[GET] ${rel}"
  if [[ -s "${part}" ]]; then
    echo "[INFO] Resuming partial file: ${part}"
  fi

  if ! curl -L --fail --retry 8 --retry-delay 2 --connect-timeout 20 \
      --progress-bar -C - -o "${part}" "${url}"; then
    echo "[WARN] Resume failed for ${rel}; retrying once from byte 0."
    rm -f "${part}"
    curl -L --fail --retry 8 --retry-delay 2 --connect-timeout 20 \
      --progress-bar -o "${part}" "${url}"
  fi

  if [[ ! -s "${part}" ]]; then
    echo "[ERROR] Download produced an empty file: ${rel}" >&2
    exit 3
  fi

  mv "${part}" "${dest}"
  echo "[OK] ${rel} ($(du -h "${dest}" | cut -f1))"
}

echo
echo '[2/3] Downloading/resuming files via direct GET...'
while IFS= read -r file; do
  [[ -z "${file}" ]] && continue
  download_one "${file}"
done < "${TMP_LIST}"

echo
echo '[3/3] Verifying Monet-7B checkpoint structure...'
python - <<PY
from pathlib import Path
import json

model_dir = Path(r"${MODEL_DIR}")
required = [
    'config.json',
    'generation_config.json',
    'model.safetensors.index.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'preprocessor_config.json',
]
missing = [x for x in required if not (model_dir / x).is_file()]
if missing:
    raise SystemExit('Missing required files: ' + ', '.join(missing))

with open(model_dir / 'model.safetensors.index.json', 'r', encoding='utf-8') as f:
    index = json.load(f)

expected_shards = sorted(set(index.get('weight_map', {}).values()))
if not expected_shards:
    raise SystemExit('No shard names found in model.safetensors.index.json')

missing_shards = [x for x in expected_shards if not (model_dir / x).is_file()]
if missing_shards:
    raise SystemExit('Missing model shards: ' + ', '.join(missing_shards))

small_shards = [x for x in expected_shards if (model_dir / x).stat().st_size < 100_000_000]
if small_shards:
    raise SystemExit('Suspiciously small model shards: ' + ', '.join(small_shards))

with open(model_dir / 'config.json', 'r', encoding='utf-8') as f:
    cfg = json.load(f)

actual_bytes = sum((model_dir / x).stat().st_size for x in expected_shards)
tensor_bytes = int(index.get('metadata', {}).get('total_size', 0) or 0)
if tensor_bytes and actual_bytes < tensor_bytes:
    raise SystemExit(f'Shard bytes ({actual_bytes}) are smaller than indexed tensor bytes ({tensor_bytes}).')

print('model_type:', cfg.get('model_type'))
print('architectures:', cfg.get('architectures'))
print('expected shards:', len(expected_shards))
for shard in expected_shards:
    print('  ', shard, round((model_dir / shard).stat().st_size / 1024**3, 3), 'GiB')
print('indexed tensor size GiB:', round(tensor_bytes / 1024**3, 3) if tensor_bytes else 'n/a')
print('actual shard size GiB:', round(actual_bytes / 1024**3, 3))
print('model directory:', model_dir)
PY

du -sh "${MODEL_DIR}" | tee "${ROOT_DIR}/logs/03_monet7b_size.txt"
find "${MODEL_DIR}" -maxdepth 1 -type f -printf '%f\n' | sort | head -100

echo
printf '%s\n' '============================================================'
echo '[DONE] Monet-7B direct-GET download and verification completed.'
echo 'Next: run scripts/04_monet_smoke_inference.sh.'
printf '%s\n' '============================================================'
