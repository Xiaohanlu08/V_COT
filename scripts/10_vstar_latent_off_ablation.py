#!/usr/bin/env python3
import json
import math
import os
import re
from pathlib import Path
from statistics import mean

LATENT_START_ID = 151666
LATENT_END_ID = 151667
LATENT_START_TOKEN = "<abs_vis_token>"
OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. You are required to answer the question "
    "based on the image provided. Put your final answer in \\boxed{}."
)


def extract_option(text):
    # Diagnostic only; this is not the official VLMEvalKit judge.
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    candidates = boxed[-1:] if boxed else [text]
    for candidate in candidates:
        m = re.search(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", candidate.upper())
        if m:
            return m.group(1)
    return None


def mcnemar_exact_p(b, c):
    """Exact two-sided McNemar p using Binomial(n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main():
    from importlib.metadata import version

    import torch
    import vllm.v1.worker.gpu_model_runner as runner
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    root_dir = Path(os.environ["VCOT_ROOT"]).resolve()
    model_dir = (root_dir / "models" / "Monet-7B").resolve()
    baseline_jsonl = Path(os.environ["VCOT_BASELINE_JSONL"]).resolve()
    out_jsonl = Path(os.environ["VCOT_OUT_JSONL"]).resolve()
    out_summary = Path(os.environ["VCOT_OUT_SUMMARY"]).resolve()
    n_requested = int(os.environ.get("VCOT_N", "5"))

    if not baseline_jsonl.is_file():
        raise FileNotFoundError(f"Baseline JSONL missing: {baseline_jsonl}")

    baseline_records = [
        json.loads(line)
        for line in baseline_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    triggered_baseline = sorted(
        [r for r in baseline_records if r.get("triggered")],
        key=lambda r: r["dataset_position"],
    )

    if not triggered_baseline:
        raise RuntimeError("Baseline JSONL contains no triggered samples.")

    if n_requested == 0:
        selected = triggered_baseline
    else:
        if n_requested < 1 or n_requested > len(triggered_baseline):
            raise ValueError(
                f"VCOT_N must be 0 (all) or in [1, {len(triggered_baseline)}], "
                f"got {n_requested}"
            )
        selected = triggered_baseline[:n_requested]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("", encoding="utf-8")

    print("========== ENVIRONMENT ==========")
    print("VCOT_ROOT:", root_dir)
    print("MODEL_DIR:", model_dir)
    print("BASELINE_JSONL:", baseline_jsonl)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", version("torch"))
    print("transformers:", version("transformers"))
    print("vllm:", version("vllm"))
    print("LATENT_SIZE:", os.environ.get("LATENT_SIZE"))
    print("baseline triggered samples:", len(triggered_baseline))
    print("selected intervention samples:", len(selected))
    print("selected positions:", [r["dataset_position"] for r in selected])

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
    print("dataset class:", type(dataset).__name__)
    print("dataset name:", getattr(dataset, "dataset_name", "unknown"))
    print("num samples:", len(dataset.data))

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
    start_id = tokenizer.convert_tokens_to_ids(LATENT_START_TOKEN)
    end_id = tokenizer.convert_tokens_to_ids("</abs_vis_token>")
    print("tokenizer start id:", start_id)
    print("tokenizer end id:", end_id)
    if start_id != LATENT_START_ID or end_id != LATENT_END_ID:
        raise RuntimeError(f"Unexpected latent token IDs: start={start_id}, end={end_id}")

    last_capture = {}
    original_generate = model.llm.generate

    def latent_off_generate(*args, **kwargs):
        last_capture.clear()
        if args:
            last_capture["request"] = args[0]
        else:
            last_capture["request"] = kwargs.get("prompts")

        sampling_params = kwargs.get("sampling_params")
        if sampling_params is None:
            raise RuntimeError("Expected sampling_params in Qwen2VLChat vLLM path.")

        if getattr(sampling_params, "temperature", None) != 0.0:
            raise RuntimeError(
                f"Expected greedy temperature=0.0, got {sampling_params.temperature}"
            )
        if getattr(sampling_params, "allowed_token_ids", None) not in (None, []):
            raise RuntimeError(
                "Unexpected pre-existing allowed_token_ids constraint: "
                f"{sampling_params.allowed_token_ids}"
            )

        # Causal intervention: block ONLY the latent-start special token.
        # vLLM 0.10.0 converts bad_words to token IDs via update_from_tokenizer.
        sampling_params.bad_words = [LATENT_START_TOKEN]
        sampling_params.update_from_tokenizer(tokenizer)
        bad_ids = getattr(sampling_params, "_bad_words_token_ids", None)
        last_capture["bad_words_token_ids"] = bad_ids
        last_capture["sampling_params"] = sampling_params

        if not bad_ids or [LATENT_START_ID] not in bad_ids:
            raise RuntimeError(
                "Latent-start block was not tokenized to the expected singleton ID. "
                f"Observed _bad_words_token_ids={bad_ids}"
            )

        outputs = original_generate(*args, **kwargs)
        last_capture["outputs"] = outputs
        return outputs

    model.llm.generate = latent_off_generate

    records = []
    print("\n========== PAIRED LATENT-OFF ABLATION ==========")

    for ordinal, baseline in enumerate(selected, start=1):
        pos = int(baseline["dataset_position"])
        row = dataset.data.iloc[pos]
        prompt = dataset.build_prompt(pos)

        image_items = [x for x in prompt if x.get("type") == "image"]
        if not image_items:
            raise RuntimeError(f"No image in prompt at dataset position {pos}")
        for item in image_items:
            if not Path(item["value"]).is_file():
                raise FileNotFoundError(item["value"])

        returned_text = model.generate(prompt, dataset="VStarBench")
        outputs = last_capture.get("outputs")
        if outputs is None or len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(f"Unexpected vLLM output shape at position {pos}")

        candidate = outputs[0].outputs[0]
        token_ids = list(candidate.token_ids)
        raw_text = candidate.text

        # The intervention is only valid if latent start never appears.
        residual_start_positions = [
            i for i, token_id in enumerate(token_ids)
            if token_id == LATENT_START_ID
        ]
        if residual_start_positions:
            raise RuntimeError(
                f"Latent-start token escaped block at pos={pos}: "
                f"{residual_start_positions}"
            )

        pred_off = extract_option(raw_text)
        gt = str(row.get("answer", "")).strip().upper()
        correct_off = pred_off == gt if pred_off is not None else False
        baseline_correct = bool(baseline["heuristic_option_correct"])

        if baseline_correct and correct_off:
            transition = "correct->correct"
        elif baseline_correct and not correct_off:
            transition = "correct->wrong"
        elif not baseline_correct and correct_off:
            transition = "wrong->correct"
        else:
            transition = "wrong->wrong"

        record = {
            "ordinal": ordinal,
            "dataset_position": pos,
            "benchmark_index": baseline.get("benchmark_index"),
            "category": baseline.get("category"),
            "ground_truth": gt,
            "baseline_triggered": bool(baseline.get("triggered")),
            "baseline_predicted_option_diagnostic": baseline.get("predicted_option_diagnostic"),
            "baseline_heuristic_correct": baseline_correct,
            "baseline_num_generated_tokens": baseline.get("num_generated_tokens"),
            "baseline_raw_text": baseline.get("raw_text"),
            "latent_off_predicted_option_diagnostic": pred_off,
            "latent_off_heuristic_correct": correct_off,
            "latent_off_num_generated_tokens": len(token_ids),
            "latent_off_raw_text": raw_text,
            "latent_off_returned_text": returned_text,
            "latent_off_token_ids": token_ids,
            "residual_latent_start_positions": residual_start_positions,
            "bad_words_token_ids": last_capture.get("bad_words_token_ids"),
            "transition": transition,
        }
        records.append(record)

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[{ordinal:02d}/{len(selected):02d}] "
            f"pos={pos:03d} cat={str(record['category']):<20s} "
            f"base={record['baseline_predicted_option_diagnostic']} "
            f"off={pred_off} gt={gt} transition={transition} "
            f"tokens={record['baseline_num_generated_tokens']}->{len(token_ids)}"
        )

    counts = {
        key: sum(r["transition"] == key for r in records)
        for key in [
            "correct->correct",
            "correct->wrong",
            "wrong->correct",
            "wrong->wrong",
        ]
    }
    baseline_correct_count = sum(r["baseline_heuristic_correct"] for r in records)
    latent_off_correct_count = sum(r["latent_off_heuristic_correct"] for r in records)
    harm = counts["correct->wrong"]
    help_ = counts["wrong->correct"]

    summary = {
        "dataset": "VStarBench",
        "intervention": "block only <abs_vis_token> using vLLM bad_words",
        "selected_from": str(baseline_jsonl),
        "baseline_triggered_total": len(triggered_baseline),
        "n_intervened": len(records),
        "selected_positions": [r["dataset_position"] for r in records],
        "latent_size": 10,
        "latent_start_id": LATENT_START_ID,
        "block_verified_samples": sum(
            len(r["residual_latent_start_positions"]) == 0 for r in records
        ),
        "baseline_heuristic_correct_count": baseline_correct_count,
        "baseline_heuristic_accuracy": baseline_correct_count / len(records),
        "latent_off_heuristic_correct_count": latent_off_correct_count,
        "latent_off_heuristic_accuracy": latent_off_correct_count / len(records),
        "accuracy_delta_latent_off_minus_baseline": (
            latent_off_correct_count - baseline_correct_count
        ) / len(records),
        "transitions": counts,
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(harm, help_),
        "mean_tokens_baseline": mean(r["baseline_num_generated_tokens"] for r in records),
        "mean_tokens_latent_off": mean(r["latent_off_num_generated_tokens"] for r in records),
        "note": (
            "Correctness uses the same diagnostic option parser as Step 09. "
            "This paired ablation is causal with respect to blocking the latent-start token "
            "under the tested decoding path, but it is not the official supplementary-judge score."
        ),
    }

    out_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n========== SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nVSTAR_LATENT_OFF_ABLATION_PASS=True")


if __name__ == "__main__":
    main()
