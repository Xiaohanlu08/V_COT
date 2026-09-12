#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONET_DIR="${ROOT_DIR}/third_party/Monet"
EXPECTED_MONET_SHA="08939998d3d643a73a316e349faa34f420429153"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vcot

if [[ ! -d "${MONET_DIR}/.git" ]]; then
  echo "[ERROR] Monet source tree not found: ${MONET_DIR}" >&2
  exit 1
fi

ACTUAL_SHA="$(git -C "${MONET_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_MONET_SHA}" ]]; then
  echo "[ERROR] Monet commit mismatch." >&2
  echo "expected: ${EXPECTED_MONET_SHA}" >&2
  echo "actual  : ${ACTUAL_SHA}" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vcot_monet_official_runner.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${WORK_DIR}/Monet_models"
cp "${MONET_DIR}/inference/vllm/monet_gpu_model_runner.py" \
   "${WORK_DIR}/Monet_models/monet_gpu_model_runner.py"

# Monet README at the pinned commit says `sitecustomized.py`, while the very
# next comment says `sitecustomize.py`. Python's automatic startup hook is
# `sitecustomize.py`, so this check corrects only that filename. The Python
# body below is otherwise the official README patch logic.
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

cat > "${WORK_DIR}/spawn_check.py" <<'PYCODE'
import multiprocessing as mp
import os
from pathlib import Path

EXPECTED = (
    Path(os.environ["VCOT_OFFICIAL_CHECK_DIR"])
    / "Monet_models"
    / "monet_gpu_model_runner.py"
).resolve()


def inspect_runner(label: str) -> None:
    import vllm.v1.worker.gpu_model_runner as runner

    actual = Path(runner.__file__).resolve()
    print(f"{label}_pid={os.getpid()}", flush=True)
    print(f"{label}_runner_file={actual}", flush=True)
    print(f"{label}_runner_class_module={runner.GPUModelRunner.__module__}", flush=True)

    if actual != EXPECTED:
        raise RuntimeError(
            f"{label}: expected Monet runner {EXPECTED}, but loaded {actual}"
        )


def child_main() -> None:
    inspect_runner("CHILD")


def main() -> None:
    print("=============== OFFICIAL PATCH CHECK ===============", flush=True)
    print(f"check_dir={Path(os.environ['VCOT_OFFICIAL_CHECK_DIR']).resolve()}", flush=True)
    print(f"expected_runner={EXPECTED}", flush=True)
    print(f"LATENT_START_ID={os.environ.get('LATENT_START_ID')}", flush=True)
    print(f"LATENT_END_ID={os.environ.get('LATENT_END_ID')}", flush=True)

    inspect_runner("PARENT")

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=child_main)
    proc.start()
    proc.join()

    print(f"CHILD_exitcode={proc.exitcode}", flush=True)
    if proc.exitcode != 0:
        raise SystemExit(proc.exitcode)

    print("OFFICIAL_RUNNER_PATCH_PASS=True", flush=True)


if __name__ == "__main__":
    main()
PYCODE

export VCOT_OFFICIAL_CHECK_DIR="${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

echo
echo "[V_COT] Monet official-style spawned-runner check"
echo "[V_COT] Monet commit: ${ACTUAL_SHA}"
echo "[V_COT] temporary check dir: ${WORK_DIR}"
echo "[V_COT] No model weights will be loaded."
echo

cd "${WORK_DIR}"
python spawn_check.py
