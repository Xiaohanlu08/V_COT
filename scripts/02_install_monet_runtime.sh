#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
EXPECTED_MONET_SHA="08939998d3d643a73a316e349faa34f420429153"
PIP_INDEX="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

mkdir -p "${ROOT_DIR}/logs"

printf '\n[V_COT] Installing pinned Monet runtime\n'
printf '[V_COT] root: %s\n' "${ROOT_DIR}"
printf '[V_COT] pip mirror: %s\n\n' "${PIP_INDEX}"

if ! command -v conda >/dev/null 2>&1; then
  echo '[ERROR] conda was not found in PATH.' >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx 'vcot'; then
  echo '[ERROR] conda env vcot does not exist. Run scripts/01_prepare_restricted_env.sh first.' >&2
  exit 2
fi

if [[ ! -d "${MONET_DIR}/.git" ]]; then
  echo "[ERROR] Monet git tree missing at ${MONET_DIR}" >&2
  exit 3
fi

ACTUAL_MONET_SHA="$(git -C "${MONET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_MONET_SHA}" != "${EXPECTED_MONET_SHA}" ]]; then
  echo "[ERROR] Monet SHA mismatch." >&2
  echo "Expected: ${EXPECTED_MONET_SHA}" >&2
  echo "Actual  : ${ACTUAL_MONET_SHA}" >&2
  exit 4
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

# DeepSpeed should install without attempting to compile CUDA extensions.
# CUDA extension compilation is deferred until/if the later training path needs it.
export DS_BUILD_OPS=0
export PIP_INDEX_URL="${PIP_INDEX}"

printf '%s\n' '================ PRE-INSTALL ================'
python --version
python -m pip --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
df -h "${ROOT_DIR}" || true

printf '\n[1/4] Installing the version-critical Monet/vLLM stack...\n'
python -m pip install --prefer-binary -i "${PIP_INDEX}" \
  'torch==2.7.1' \
  'torchvision==0.22.1' \
  'vllm==0.10.0' \
  'transformers==4.54.0' \
  'trl==0.15.2'

printf '\n[2/4] Installing the remaining Monet requirements...\n'
TMP_REQ="$(mktemp)"
trap 'rm -f "${TMP_REQ}"' EXIT

# Exclude packages pinned above so the generic requirements file cannot silently
# replace them with newer versions.
grep -Ev '^[[:space:]]*(torch|torchvision|vllm|transformers|trl)([<>=!~[:space:]]|$)' \
  "${MONET_DIR}/requirements.txt" > "${TMP_REQ}"

python -m pip install --prefer-binary -i "${PIP_INDEX}" -r "${TMP_REQ}"

printf '\n[3/4] Running dependency consistency check...\n'
python -m pip check

printf '\n[4/4] Running CUDA/runtime probe...\n'
python - <<'PY'
import torch
import torchvision
import transformers
import trl
import vllm

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('transformers:', transformers.__version__)
print('trl:', trl.__version__)
print('vllm:', vllm.__version__)
print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())

assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.22.1', torchvision.__version__
assert transformers.__version__ == '4.54.0', transformers.__version__
assert trl.__version__ == '0.15.2', trl.__version__
assert vllm.__version__ == '0.10.0', vllm.__version__
assert torch.cuda.is_available(), 'CUDA is not available in PyTorch'

# Minimal CUDA execution test on physical GPU 0. This does not load Monet.
x = torch.tensor([1.0, 2.0, 3.0], device='cuda:0')
y = (x * 2).sum()
print('CUDA tensor test:', y.item())
print('GPU0:', torch.cuda.get_device_name(0))
print('GPU0 capability:', torch.cuda.get_device_capability(0))
PY

printf '\n%s\n' '================ PINNED VERSIONS ================'
python -m pip show torch torchvision vllm transformers trl | \
  grep -E '^(Name|Version):' || true

cat <<'EOF'

============================================================
[DONE] Monet runtime installation completed.

Notes:
- Absence of system nvcc is not a blocker for this prebuilt-wheel inference/runtime gate.
- Do NOT install another torch/vLLM/transformers version after this step.
- Do NOT start Monet inference yet.
- Next gate: download Monet-7B through hf-mirror and run a minimal load/inference test.
============================================================
EOF
