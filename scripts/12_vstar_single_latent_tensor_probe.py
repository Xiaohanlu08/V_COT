#!/usr/bin/env python3
import json
import os
from pathlib import Path

LATENT_START_ID = 151666
LATENT_END_ID = 151667
OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. You are required to answer the question "
    "based on the image provided. Put your final answer in \\boxed{}."
)


def main():
    from importlib.metadata import version

    import torch
    import torch.nn.functional as F
    import vllm.v1.worker.gpu_model_runner as runner
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    root = Path(os.environ["VCOT_ROOT"]).resolve()
    model_dir = root / "models" / "Monet-7B"
    baseline_jsonl = root / "results" / "natural_trigger" / "vstar_n191_seed20260913.jsonl"
    dump_dir = Path(os.environ["VCOT_LATENT_DUMP_DIR"]).resolve()
    result_dir = root / "results" / "latent_capture"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("========== ENVIRONMENT ==========")
    print("VCOT_ROOT:", root)
    print("MODEL_DIR:", model_dir)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", version("torch"))
    print("transformers:", version("transformers"))
    print("vllm:", version("vllm"))
    print("LATENT_SIZE:", os.environ.get("LATENT_SIZE"))
    print("runner_file:", Path(runner.__file__).resolve())
    print("dump_dir:", dump_dir)

    if Path(runner.__file__).name != "monet_gpu_model_runner.py":
        raise RuntimeError("Instrumented Monet runner is not active.")
    if os.environ.get("LATENT_SIZE") != "10":
        raise RuntimeError("Expected LATENT_SIZE=10")
    if os.environ.get("VCOT_LATENT_TENSOR_DUMP") != "1":
        raise RuntimeError("VCOT_LATENT_TENSOR_DUMP must equal 1")
    if not baseline_jsonl.is_file():
        raise FileNotFoundError(baseline_jsonl)

    baseline_records = [
        json.loads(line)
        for line in baseline_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = next(r for r in baseline_records if int(r["dataset_position"]) == 0)
    if not baseline.get("triggered"):
        raise RuntimeError("Expected VStarBench position 0 to be naturally triggered in baseline JSONL.")

    dataset = build_dataset("VStarBench")
    prompt = dataset.build_prompt(0)

    model = Qwen2VLChat(
        model_path=str(model_dir),
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_new_tokens=2048,
        use_custom_prompt=False,
        system_prompt=OFFICIAL_SYSTEM_PROMPT,
        post_process=False,
        verbose=False,
        use_vllm=True,
    )

    tokenizer = model.processor.tokenizer
    start_id = tokenizer.convert_tokens_to_ids("<abs_vis_token>")
    end_id = tokenizer.convert_tokens_to_ids("</abs_vis_token>")
    print("tokenizer start id:", start_id)
    print("tokenizer end id:", end_id)
    if start_id != LATENT_START_ID or end_id != LATENT_END_ID:
        raise RuntimeError("Unexpected latent token IDs")

    captured = {}
    original_generate = model.llm.generate

    def capture_generate(*args, **kwargs):
        outputs = original_generate(*args, **kwargs)
        captured["outputs"] = outputs
        captured["sampling_params"] = kwargs.get("sampling_params")
        return outputs

    model.llm.generate = capture_generate

    print("\n========== GENERATE POSITION 0 ==========")
    returned_text = model.generate(prompt, dataset="VStarBench")

    outputs = captured.get("outputs")
    if outputs is None or len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("Unexpected vLLM output shape")

    candidate = outputs[0].outputs[0]
    token_ids = list(candidate.token_ids)
    raw_text = candidate.text

    baseline_token_ids = list(baseline["token_ids"])
    generation_exact_match = token_ids == baseline_token_ids

    print("generated token count:", len(token_ids))
    print("baseline token count:", len(baseline_token_ids))
    print("generation_exact_match_baseline:", generation_exact_match)
    print("start positions:", [i for i, x in enumerate(token_ids) if x == LATENT_START_ID])
    print("end positions:", [i for i, x in enumerate(token_ids) if x == LATENT_END_ID])

    if not generation_exact_match:
        raise RuntimeError(
            "Tensor instrumentation changed greedy generation; refusing to treat capture as observation-only."
        )
    if returned_text != raw_text:
        raise RuntimeError("VLMEval returned text differs from raw vLLM text")

    files = sorted(dump_dir.glob("latent_step_*.pt"))
    print("\n========== LATENT TENSOR FILES ==========")
    print("num_files:", len(files))
    for f in files:
        print(f.name)

    if len(files) != 10:
        raise RuntimeError(f"Expected 10 latent tensors, found {len(files)}")

    tensors = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    if any(t.ndim != 1 for t in tensors):
        raise RuntimeError(f"Expected all tensors to be 1D; shapes={[tuple(t.shape) for t in tensors]}")
    hidden_sizes = {int(t.numel()) for t in tensors}
    if len(hidden_sizes) != 1:
        raise RuntimeError(f"Latent hidden sizes disagree: {hidden_sizes}")
    if any(not torch.isfinite(t).all().item() for t in tensors):
        raise RuntimeError("Non-finite values detected in latent tensors")

    z = torch.stack(tensors, dim=0).float()
    norms = z.norm(dim=1)
    adjacent_cos = [
        F.cosine_similarity(z[i:i+1], z[i+1:i+2], dim=1).item()
        for i in range(z.shape[0] - 1)
    ]

    combined_path = result_dir / "vstar_pos0_latents.pt"
    torch.save(
        {
            "dataset": "VStarBench",
            "dataset_position": 0,
            "latent_size": 10,
            "latent_start_id": LATENT_START_ID,
            "latent_end_id": LATENT_END_ID,
            "tensor": z,
            "token_ids": token_ids,
            "raw_text": raw_text,
        },
        combined_path,
    )

    summary = {
        "dataset": "VStarBench",
        "dataset_position": 0,
        "generation_exact_match_baseline": generation_exact_match,
        "num_latent_tensors": int(z.shape[0]),
        "hidden_size": int(z.shape[1]),
        "dtype_saved": str(z.dtype),
        "all_finite": bool(torch.isfinite(z).all().item()),
        "mean_l2_norm": float(norms.mean().item()),
        "min_l2_norm": float(norms.min().item()),
        "max_l2_norm": float(norms.max().item()),
        "adjacent_cosine_mean": float(sum(adjacent_cos) / len(adjacent_cos)),
        "adjacent_cosine_min": float(min(adjacent_cos)),
        "adjacent_cosine_max": float(max(adjacent_cos)),
        "combined_tensor_path": str(combined_path),
    }
    summary_path = result_dir / "vstar_pos0_latents_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n========== LATENT TENSOR SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("\nVSTAR_SINGLE_LATENT_TENSOR_PROBE_PASS=True")


if __name__ == "__main__":
    main()
