#!/usr/bin/env python3
import json
import os
import random
import re
from pathlib import Path

LATENT_START_ID = 151666
LATENT_END_ID = 151667
OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. You are required to answer the question "
    "based on the image provided. Put your final answer in \\boxed{}."
)


def find_latent_segments(token_ids):
    segments = []
    active_start = None
    for i, token_id in enumerate(token_ids):
        if token_id == LATENT_START_ID and active_start is None:
            active_start = i
        elif token_id == LATENT_END_ID and active_start is not None:
            segments.append((active_start, i))
            active_start = None
    if active_start is not None:
        segments.append((active_start, None))
    return segments


def extract_option(text):
    # Diagnostic only; this is not the official VLMEvalKit judge.
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    candidates = boxed[-1:] if boxed else [text]
    for candidate in candidates:
        m = re.search(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", candidate.upper())
        if m:
            return m.group(1)
    return None


def as_jsonable_segments(segments):
    return [[start, end] for start, end in segments]


def main():
    from importlib.metadata import version

    import torch
    import vllm.v1.worker.gpu_model_runner as runner
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    root_dir = Path(os.environ["VCOT_ROOT"]).resolve()
    model_dir = (root_dir / "models" / "Monet-7B").resolve()
    n_requested = int(os.environ.get("VCOT_N", "20"))
    seed = int(os.environ.get("VCOT_SEED", "20260913"))
    out_jsonl = Path(os.environ["VCOT_OUT_JSONL"]).resolve()
    out_summary = Path(os.environ["VCOT_OUT_SUMMARY"]).resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    print("========== ENVIRONMENT ==========")
    print("VCOT_ROOT:", root_dir)
    print("MODEL_DIR:", model_dir)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", version("torch"))
    print("transformers:", version("transformers"))
    print("vllm:", version("vllm"))
    print("LATENT_SIZE:", os.environ.get("LATENT_SIZE"))
    print("VCOT_N:", n_requested)
    print("VCOT_SEED:", seed)
    print("output jsonl:", out_jsonl)
    print("output summary:", out_summary)

    runner_file = Path(runner.__file__).resolve()
    print("runner_file:", runner_file)
    print("runner_class_module:", runner.GPUModelRunner.__module__)
    if runner_file.name != "monet_gpu_model_runner.py":
        raise RuntimeError(f"Monet runner patch is not active: {runner_file}")
    if os.environ.get("LATENT_SIZE") != "10":
        raise RuntimeError(f"Expected LATENT_SIZE=10, got {os.environ.get('LATENT_SIZE')}")

    torch.set_grad_enabled(False)

    print("\n========== BUILD VSTAR DATASET ==========")
    dataset = build_dataset("VStarBench")
    total = len(dataset.data)
    if n_requested < 1 or n_requested > total:
        raise ValueError(f"VCOT_N must be in [1, {total}], got {n_requested}")
    rng = random.Random(seed)
    sample_positions = sorted(rng.sample(range(total), n_requested))
    print("dataset class:", type(dataset).__name__)
    print("dataset name:", getattr(dataset, "dataset_name", "unknown"))
    print("num samples:", total)
    print("selected positions:", sample_positions)

    print("\n========== INITIALIZE MONET VIA VLMEVALKIT ==========")
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
        raise RuntimeError(f"Unexpected latent token IDs: start={start_id}, end={end_id}")

    last_capture = {}
    original_generate = model.llm.generate

    def capture_generate(*args, **kwargs):
        last_capture.clear()
        if args:
            last_capture["request"] = args[0]
        else:
            last_capture["request"] = kwargs.get("prompts")
        last_capture["sampling_params"] = kwargs.get("sampling_params")
        outputs = original_generate(*args, **kwargs)
        last_capture["outputs"] = outputs
        return outputs

    model.llm.generate = capture_generate

    records = []
    out_jsonl.write_text("", encoding="utf-8")

    print("\n========== NATURAL TRIGGER SCAN ==========")
    for ordinal, pos in enumerate(sample_positions, start=1):
        row = dataset.data.iloc[pos]
        prompt = dataset.build_prompt(pos)

        image_items = [x for x in prompt if x.get("type") == "image"]
        text_items = [x for x in prompt if x.get("type") == "text"]
        if not image_items or not text_items:
            raise RuntimeError(f"Malformed prompt at dataset position {pos}: {prompt}")
        for image_item in image_items:
            image_path = Path(image_item["value"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Image missing at position {pos}: {image_path}")

        returned_text = model.generate(prompt, dataset="VStarBench")
        if "outputs" not in last_capture:
            raise RuntimeError(f"Raw vLLM output not captured at position {pos}")

        outputs = last_capture["outputs"]
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(
                f"Unexpected vLLM output shape at position {pos}: "
                f"requests={len(outputs)}, candidates={len(outputs[0].outputs) if outputs else 'n/a'}"
            )
        candidate = outputs[0].outputs[0]
        token_ids = list(candidate.token_ids)
        raw_text = candidate.text

        sampling_params = last_capture.get("sampling_params")
        temperature = getattr(sampling_params, "temperature", None)
        max_tokens = getattr(sampling_params, "max_tokens", None)
        stop_token_ids = getattr(sampling_params, "stop_token_ids", None)
        allowed_token_ids = getattr(sampling_params, "allowed_token_ids", None)
        if temperature != 0.0:
            raise RuntimeError(f"Expected temperature=0.0 at position {pos}, got {temperature}")
        if allowed_token_ids not in (None, []):
            raise RuntimeError(
                f"Forced-token constraint detected at position {pos}: {allowed_token_ids}"
            )

        request = last_capture.get("request")
        final_prompt = request.get("prompt") if isinstance(request, dict) else None
        if final_prompt is not None and OFFICIAL_SYSTEM_PROMPT not in final_prompt:
            raise RuntimeError(f"Official Monet system prompt missing at position {pos}")

        start_positions = [i for i, t in enumerate(token_ids) if t == LATENT_START_ID]
        end_positions = [i for i, t in enumerate(token_ids) if t == LATENT_END_ID]
        segments = find_latent_segments(token_ids)
        triggered = bool(start_positions)
        complete_segments = sum(1 for _, end in segments if end is not None)
        predicted_option = extract_option(raw_text)
        gt = str(row.get("answer", "")).strip().upper()
        heuristic_correct = predicted_option == gt if predicted_option is not None else False

        record = {
            "ordinal": ordinal,
            "dataset_position": pos,
            "benchmark_index": int(row["index"]) if str(row["index"]).isdigit() else str(row["index"]),
            "category": None if "category" not in row else str(row["category"]),
            "ground_truth": gt,
            "predicted_option_diagnostic": predicted_option,
            "heuristic_option_correct": heuristic_correct,
            "num_generated_tokens": len(token_ids),
            "token_ids": token_ids,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "segments": as_jsonable_segments(segments),
            "num_start_tokens": len(start_positions),
            "num_end_tokens": len(end_positions),
            "num_latent_segments": len(segments),
            "num_complete_latent_segments": complete_segments,
            "triggered": triggered,
            "raw_text": raw_text,
            "returned_text": returned_text,
            "returned_text_matches_raw_text": returned_text == raw_text,
            "dataset_prompt_text": "\n".join(x["value"] for x in text_items),
            "final_chat_template_prompt": final_prompt,
            "sampling": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop_token_ids": stop_token_ids,
                "allowed_token_ids": allowed_token_ids,
            },
        }
        records.append(record)
        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[{ordinal:02d}/{n_requested:02d}] "
            f"pos={pos:03d} idx={record['benchmark_index']} "
            f"trigger={triggered} segments={len(segments)} "
            f"tokens={len(token_ids)} gt={gt} pred={predicted_option}"
        )

    triggered_records = [r for r in records if r["triggered"]]
    balanced_records = [
        r for r in records
        if r["num_start_tokens"] == r["num_end_tokens"]
        and r["num_start_tokens"] == r["num_complete_latent_segments"]
    ]
    multi_segment_records = [r for r in records if r["num_latent_segments"] > 1]
    heuristic_correct_count = sum(1 for r in records if r["heuristic_option_correct"])

    summary = {
        "dataset": "VStarBench",
        "n": len(records),
        "seed": seed,
        "sample_positions": sample_positions,
        "latent_size": 10,
        "latent_start_id": LATENT_START_ID,
        "latent_end_id": LATENT_END_ID,
        "triggered_samples": len(triggered_records),
        "trigger_rate": len(triggered_records) / len(records),
        "balanced_marker_samples": len(balanced_records),
        "multi_segment_samples": len(multi_segment_records),
        "total_latent_segments": sum(r["num_latent_segments"] for r in records),
        "mean_generated_tokens": sum(r["num_generated_tokens"] for r in records) / len(records),
        "heuristic_option_correct_count": heuristic_correct_count,
        "heuristic_option_accuracy": heuristic_correct_count / len(records),
        "note": "Heuristic option accuracy is diagnostic only and is not the official Monet/VLMEvalKit benchmark score.",
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nVSTAR_NATURAL_TRIGGER_SCAN_PASS=True")


if __name__ == "__main__":
    main()
