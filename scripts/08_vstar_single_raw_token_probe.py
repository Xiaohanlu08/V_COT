#!/usr/bin/env python3
import os
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


def print_token_ids(token_ids):
    if len(token_ids) <= 512:
        print(token_ids)
        return

    print(token_ids[:256])
    print(f"... <{len(token_ids) - 512} token ids omitted> ...")
    print(token_ids[-256:])


def main():
    from importlib.metadata import version

    import torch
    import vllm.v1.worker.gpu_model_runner as runner
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    root_dir = Path(os.environ["VCOT_ROOT"]).resolve()
    model_dir = (root_dir / "models" / "Monet-7B").resolve()

    print("========== ENVIRONMENT ==========")
    print("VCOT_ROOT:", root_dir)
    print("MODEL_DIR:", model_dir)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", version("torch"))
    print("transformers:", version("transformers"))
    print("vllm:", version("vllm"))
    print("LATENT_START_ID:", os.environ.get("LATENT_START_ID"))
    print("LATENT_END_ID:", os.environ.get("LATENT_END_ID"))
    print("LATENT_SIZE:", os.environ.get("LATENT_SIZE"))

    runner_file = Path(runner.__file__).resolve()
    print("runner_file:", runner_file)
    print("runner_class_module:", runner.GPUModelRunner.__module__)

    if runner_file.name != "monet_gpu_model_runner.py":
        raise RuntimeError(
            "Monet runner patch is not active. "
            f"Observed runner file: {runner_file}"
        )

    if os.environ.get("LATENT_SIZE") != "10":
        raise RuntimeError(
            f"Expected LATENT_SIZE=10, got {os.environ.get('LATENT_SIZE')}"
        )

    torch.set_grad_enabled(False)

    print("\n========== BUILD VSTAR DATASET ==========")
    dataset = build_dataset("VStarBench")
    print("dataset class:", type(dataset).__name__)
    print("dataset name:", getattr(dataset, "dataset_name", "unknown"))
    print("num samples:", len(dataset.data))

    row = dataset.data.iloc[0]
    prompt = dataset.build_prompt(0)

    image_items = [x for x in prompt if x.get("type") == "image"]
    text_items = [x for x in prompt if x.get("type") == "text"]

    if len(image_items) != 1:
        raise RuntimeError(f"Expected one image in sample 0, got {len(image_items)}")
    if len(text_items) != 1:
        raise RuntimeError(f"Expected one text item in sample 0, got {len(text_items)}")

    image_path = Path(image_items[0]["value"])
    if not image_path.is_file():
        raise FileNotFoundError(f"VStar image not found: {image_path}")

    print("sample index:", row["index"])
    print("ground-truth answer:", row["answer"])
    print("image path:", image_path)
    print("\n---------- DATASET PROMPT ----------")
    print(text_items[0]["value"])

    print("\n========== INITIALIZE MONET VIA VLMEVALKIT ==========")
    print("Qwen2VLChat source:", __import__(
        "vlmeval.vlm.qwen2_vl.model", fromlist=["__file__"]
    ).__file__)

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
    tokenizer_start_id = tokenizer.convert_tokens_to_ids("<abs_vis_token>")
    tokenizer_end_id = tokenizer.convert_tokens_to_ids("</abs_vis_token>")

    print("tokenizer start id:", tokenizer_start_id)
    print("tokenizer end id:", tokenizer_end_id)

    if tokenizer_start_id != LATENT_START_ID:
        raise RuntimeError(
            f"Unexpected latent start token id: {tokenizer_start_id}"
        )
    if tokenizer_end_id != LATENT_END_ID:
        raise RuntimeError(
            f"Unexpected latent end token id: {tokenizer_end_id}"
        )

    captured = {}
    original_generate = model.llm.generate

    def capture_generate(*args, **kwargs):
        if args:
            captured["request"] = args[0]
        else:
            captured["request"] = kwargs.get("prompts")

        captured["sampling_params"] = kwargs.get("sampling_params")
        outputs = original_generate(*args, **kwargs)
        captured["outputs"] = outputs
        return outputs

    model.llm.generate = capture_generate

    print("\n========== GENERATE ==========")
    returned_text = model.generate(prompt, dataset="VStarBench")

    if "outputs" not in captured:
        raise RuntimeError("Raw vLLM outputs were not captured.")

    outputs = captured["outputs"]
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one vLLM RequestOutput, got {len(outputs)}")
    if len(outputs[0].outputs) != 1:
        raise RuntimeError(
            f"Expected one completion candidate, got {len(outputs[0].outputs)}"
        )

    candidate = outputs[0].outputs[0]
    token_ids = list(candidate.token_ids)
    raw_text = candidate.text

    request = captured.get("request")
    final_prompt = None
    if isinstance(request, dict):
        final_prompt = request.get("prompt")

    sampling_params = captured.get("sampling_params")
    temperature = getattr(sampling_params, "temperature", None)
    max_tokens = getattr(sampling_params, "max_tokens", None)
    stop_token_ids = getattr(sampling_params, "stop_token_ids", None)
    allowed_token_ids = getattr(sampling_params, "allowed_token_ids", None)

    print("\n========== FINAL CHAT-TEMPLATE PROMPT ==========")
    if final_prompt is None:
        print("<not captured>")
    else:
        print(final_prompt)
        if OFFICIAL_SYSTEM_PROMPT not in final_prompt:
            raise RuntimeError("Official Monet system prompt is absent from final prompt.")

    print("\n========== SAMPLING PARAMS ==========")
    print("temperature:", temperature)
    print("max_tokens:", max_tokens)
    print("stop_token_ids:", stop_token_ids)
    print("allowed_token_ids:", allowed_token_ids)

    if temperature != 0.0:
        raise RuntimeError(f"Expected greedy temperature=0.0, got {temperature}")
    if allowed_token_ids not in (None, []):
        raise RuntimeError(
            "This diagnostic must not force token IDs. "
            f"Observed allowed_token_ids={allowed_token_ids}"
        )

    print("\n========== RAW MODEL OUTPUT ==========")
    print(raw_text)

    print("\n========== RETURNED VLMEVAL TEXT ==========")
    print(returned_text)
    print("returned_text_matches_raw_text:", returned_text == raw_text)

    print("\n========== RAW TOKEN IDS ==========")
    print("num_generated_tokens:", len(token_ids))
    print_token_ids(token_ids)

    print("\n========== TOKEN PIECES ==========")
    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    if len(token_pieces) <= 256:
        print(token_pieces)
    else:
        print(token_pieces[:128])
        print(f"... <{len(token_pieces) - 256} token pieces omitted> ...")
        print(token_pieces[-128:])

    start_positions = [
        i for i, token_id in enumerate(token_ids)
        if token_id == LATENT_START_ID
    ]
    end_positions = [
        i for i, token_id in enumerate(token_ids)
        if token_id == LATENT_END_ID
    ]
    segments = find_latent_segments(token_ids)

    print("\n========== NATURAL LATENT DIAGNOSTIC ==========")
    print("start_positions:", start_positions)
    print("end_positions:", end_positions)
    print("segments:", segments)
    print("num_start_tokens:", len(start_positions))
    print("num_end_tokens:", len(end_positions))
    print("num_latent_segments:", len(segments))
    print("NATURAL_LATENT_TRIGGER=", len(start_positions) > 0)

    print("\nVSTAR_SINGLE_RAW_TOKEN_PROBE_PASS=True")


if __name__ == "__main__":
    main()
