import importlib
import os
import sys

# Spawn-safe Monet vLLM patch.
# Python imports `sitecustomize` automatically at interpreter startup when this
# directory is on PYTHONPATH. This lets the patch run in both the parent process
# and vLLM's spawned EngineCore/worker processes.

if os.environ.get("VCOT_MONET_SITECUSTOMIZE") == "1":
    argv = " ".join(sys.argv)
    # Avoid importing the full vLLM stack inside multiprocessing's lightweight
    # resource-tracker helper process.
    if "resource_tracker" not in argv:
        os.environ.setdefault("VLLM_USE_V1", "1")
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("LATENT_START_ID", "151666")
        os.environ.setdefault("LATENT_END_ID", "151667")

        patched = importlib.import_module("inference.vllm.monet_gpu_model_runner")
        for key in (
            "vllm.v1.worker.gpu_model_runner",
            "vllm.worker.gpu_model_runner",
            "vllm.worker.model_runner",
        ):
            sys.modules[key] = patched

        print(
            f"[VCOT_MONET_SITE] pid={os.getpid()} patched Monet vLLM runner; "
            f"start={os.environ['LATENT_START_ID']} "
            f"end={os.environ['LATENT_END_ID']} "
            f"size={os.environ.get('LATENT_SIZE', '')}",
            flush=True,
        )
