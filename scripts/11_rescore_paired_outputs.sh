#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

cd "${ROOT_DIR}"
python scripts/11_rescore_paired_outputs.py \
  2>&1 | tee logs/11_rescore_paired_outputs.log
