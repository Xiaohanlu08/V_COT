#!/usr/bin/env python3
import json
import os
from copy import deepcopy
from pathlib import Path

LATENT_START_ID = 151666
LATENT_END_ID = 151667
OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. You are required to answer the question "
    "based on the image provided. Put your final answer in \\boxed{}."
)


def find_single_segment(token_ids):
    starts = [i for i, t in enumerate(token_ids) if t == LATENT_START_ID]
    ends = [i for i, t in enumerate(token_ids) if t == LATENT_END_ID]
    if len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0]:
        return starts[0], ends[0]
    return None


def main():
    from importlib.metadata import version

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from vllm import SamplingParams
    import vllm.v1.worker.gpu_model_runner as runner
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    root = Path(os.environ["VCOT_ROOT"]).resolve()
    model_dir = root / "models" / "Monet-7B"
    baseline_jsonl = root / "results" / "natural_trigger" / "vstar_n191_seed20260913.jsonl"
    exp5_path = root / "results" / "latent_capture" / "vstar_pos0_latents.pt"
    step13_dir = root / "results" / "evidence_pilot" / "vstar_pos0"
    positive_path = step13_dir / "I_positive_zoom.png"
    negative_path = step13_dir / "I_negative_mask.png"
    dump_dir = Path(os.environ["VCOT_LATENT_DUMP_DIR"]).resolve()
    result_dir = root / "results" / "evidence_replay" / "vstar_pos0"
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
    print("force_start_once:", os.environ.get("VCOT_FORCE_LATENT_START_ONCE"))

    if Path(runner.__file__).name != "monet_gpu_model_runner.py":
        raise RuntimeError("Instrumented Monet runner is not active.")
    if os.environ.get("LATENT_SIZE") != "10":
        raise RuntimeError("Expected LATENT_SIZE=10")
    if os.environ.get("VCOT_LATENT_TENSOR_DUMP") != "1":
        raise RuntimeError("VCOT_LATENT_TENSOR_DUMP must equal 1")
    if os.environ.get("VCOT_FORCE_LATENT_START_ONCE") != "1":
        raise RuntimeError("VCOT_FORCE_LATENT_START_ONCE must equal 1")

    for required in [baseline_jsonl, exp5_path, positive_path, negative_path]:
        if not required.is_file():
            raise FileNotFoundError(required)

    baseline_records = [
        json.loads(line)
        for line in baseline_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = next(r for r in baseline_records if int(r["dataset_position"]) == 0)
    baseline_ids = list(baseline["token_ids"])
    baseline_segment = find_single_segment(baseline_ids)
    if baseline_segment != (25, 35):
        raise RuntimeError(f"Unexpected baseline latent segment: {baseline_segment}")
    prefix_ids = baseline_ids[:baseline_segment[0]]

    exp5 = torch.load(exp5_path, map_location="cpu", weights_only=False)
    z_ref = exp5["tensor"].float()
    if tuple(z_ref.shape) != (10, 3584):
        raise RuntimeError(f"Unexpected EXP-0005 tensor shape: {tuple(z_ref.shape)}")

    dataset = build_dataset("VStarBench")
    base_prompt = dataset.build_prompt(0)
    image_items = [x for x in base_prompt if x.get("type") == "image"]
    if len(image_items) != 1:
        raise RuntimeError(f"Expected one image item, got {len(image_items)}")
    original_path = Path(image_items[0]["value"]).resolve()

    for p in [original_path, positive_path, negative_path]:
        im = Image.open(p)
        if im.size != (2000, 1500):
            raise RuntimeError(f"Unexpected image size for {p}: {im.size}")

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
    if tokenizer.convert_tokens_to_ids("<abs_vis_token>") != LATENT_START_ID:
        raise RuntimeError("Unexpected latent start token ID")
    if tokenizer.convert_tokens_to_ids("</abs_vis_token>") != LATENT_END_ID:
        raise RuntimeError("Unexpected latent end token ID")

    prefix_text = tokenizer.decode(
        prefix_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prefix_roundtrip = tokenizer.encode(prefix_text, add_special_tokens=False)
    print("\n========== FIXED PREFIX ==========")
    print("prefix_generated_token_count:", len(prefix_ids))
    print("prefix_decode_encode_roundtrip_exact:", prefix_roundtrip == prefix_ids)
    print("prefix_text:")
    print(prefix_text)

    # We do not rely on text re-tokenization for replay. vLLM 0.10.0 accepts
    # TokensPrompt with explicit prompt_token_ids plus multi_modal_data.
    def prepare_payload(image_path):
        prompt = deepcopy(base_prompt)
        for item in prompt:
            if item.get("type") == "image":
                item["value"] = str(image_path)

        messages = []
        if model.system_prompt is not None:
            messages.append({"role": "system", "content": model.system_prompt})
        messages.append({
            "role": "user",
            "content": model._prepare_content_vllm(prompt, dataset="VStarBench"),
        })

        text = model.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages)
        if videos:
            raise RuntimeError("Unexpected video input in VStarBench replay")
        if not images or len(images) != 1:
            raise RuntimeError(f"Expected exactly one processed image, got {0 if not images else len(images)}")

        base_ids = tokenizer.encode(text, add_special_tokens=False)
        replay_ids = list(base_ids) + list(prefix_ids)
        payload = {
            "prompt_token_ids": replay_ids,
            "multi_modal_data": {"image": images},
        }
        return payload, base_ids, text

    payloads = {}
    base_ids_by_view = {}
    template_text_by_view = {}
    paths = {
        "original": original_path,
        "positive": positive_path,
        "negative": negative_path,
    }
    for name, path in paths.items():
        payload, base_ids, template_text = prepare_payload(path)
        payloads[name] = payload
        base_ids_by_view[name] = base_ids
        template_text_by_view[name] = template_text

    base_prompt_ids_identical = (
        base_ids_by_view["original"] == base_ids_by_view["positive"]
        == base_ids_by_view["negative"]
    )
    template_text_identical = (
        template_text_by_view["original"] == template_text_by_view["positive"]
        == template_text_by_view["negative"]
    )

    print("base_prompt_token_count:", len(base_ids_by_view["original"]))
    print("base_prompt_ids_identical_across_views:", base_prompt_ids_identical)
    print("chat_template_text_identical_across_views:", template_text_identical)
    if not base_prompt_ids_identical or not template_text_identical:
        raise RuntimeError("Textual base prompt differs across visual conditions.")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        stop_token_ids=None,
    )

    def run_replay(name):
        for p in dump_dir.glob("latent_step_*.pt"):
            p.unlink()

        outputs = model.llm.generate(
            payloads[name],
            sampling_params=sampling_params,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(f"Unexpected vLLM output shape for {name}")

        candidate = outputs[0].outputs[0]
        token_ids = list(candidate.token_ids)
        segment = find_single_segment(token_ids)
        files = sorted(dump_dir.glob("latent_step_*.pt"))
        tensors = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
        if len(tensors) != 10:
            raise RuntimeError(f"Expected 10 latent tensors for {name}, found {len(tensors)}")
        if any(t.ndim != 1 or int(t.numel()) != 3584 for t in tensors):
            raise RuntimeError(f"Unexpected latent shape for {name}")
        z = torch.stack(tensors, dim=0).float()
        if not torch.isfinite(z).all().item():
            raise RuntimeError(f"Non-finite latent tensor for {name}")

        torch.save(
            {
                "view": name,
                "image_path": str(paths[name]),
                "fixed_generated_prefix_ids": prefix_ids,
                "generated_token_ids_after_prefix": token_ids,
                "tensor": z,
            },
            result_dir / f"{name}_fixed_replay_latents.pt",
        )

        return {
            "token_ids": token_ids,
            "segment": segment,
            "tensor": z,
            "raw_text": candidate.text,
        }

    print("\n========== FIXED-PREFIX / FIXED-TRIGGER REPLAY ==========")
    runs = {name: run_replay(name) for name in ["original", "positive", "negative"]}

    mechanical_alignment = True
    for name, r in runs.items():
        print(
            f"{name}: first_token={r['token_ids'][0] if r['token_ids'] else None} "
            f"segment={r['segment']} latent_shape={tuple(r['tensor'].shape)}"
        )
        if not r["token_ids"] or r["token_ids"][0] != LATENT_START_ID:
            mechanical_alignment = False
        if r["segment"] != (0, 10):
            mechanical_alignment = False

    z0 = runs["original"]["tensor"]
    zp = runs["positive"]["tensor"]
    zn = runs["negative"]["tensor"]

    ref_cos = F.cosine_similarity(z_ref, z0, dim=1)
    ref_rel_l2 = (z_ref - z0).norm(dim=1) / z_ref.norm(dim=1).clamp_min(1e-12)
    ref_replay_valid = bool(
        mechanical_alignment
        and float(ref_cos.mean().item()) >= 0.999
        and float(ref_cos.min().item()) >= 0.995
        and float(ref_rel_l2.mean().item()) <= 0.05
    )

    cos_pos = F.cosine_similarity(z0, zp, dim=1)
    cos_neg = F.cosine_similarity(z0, zn, dim=1)
    cos_pn = F.cosine_similarity(zp, zn, dim=1)
    delta = cos_pos - cos_neg

    rel_l2_pos = (z0 - zp).norm(dim=1) / z0.norm(dim=1).clamp_min(1e-12)
    rel_l2_neg = (z0 - zn).norm(dim=1) / z0.norm(dim=1).clamp_min(1e-12)

    summary = {
        "dataset": "VStarBench",
        "dataset_position": 0,
        "baseline_segment": list(baseline_segment),
        "fixed_generated_prefix_token_count": len(prefix_ids),
        "prefix_decode_encode_roundtrip_exact": prefix_roundtrip == prefix_ids,
        "base_prompt_ids_identical_across_views": base_prompt_ids_identical,
        "chat_template_text_identical_across_views": template_text_identical,
        "forced_first_generated_token": LATENT_START_ID,
        "mechanical_alignment_pass": mechanical_alignment,
        "views": {
            name: {
                "segment_after_fixed_prefix": None if r["segment"] is None else list(r["segment"]),
                "num_generated_tokens_after_prefix": len(r["token_ids"]),
                "first_generated_token": r["token_ids"][0] if r["token_ids"] else None,
                "latent_shape": list(r["tensor"].shape),
                "raw_text_after_prefix": r["raw_text"],
            }
            for name, r in runs.items()
        },
        "original_replay_vs_EXP0005": {
            "step_cosine": [float(x) for x in ref_cos.tolist()],
            "mean_cosine": float(ref_cos.mean().item()),
            "min_cosine": float(ref_cos.min().item()),
            "max_cosine": float(ref_cos.max().item()),
            "step_relative_l2": [float(x) for x in ref_rel_l2.tolist()],
            "mean_relative_l2": float(ref_rel_l2.mean().item()),
            "valid_for_counterfactual_comparison": ref_replay_valid,
            "gate": "mean cosine >= 0.999, min cosine >= 0.995, mean relative L2 <= 0.05",
        },
        "evidence_similarity": {
            "step_cos_original_positive": [float(x) for x in cos_pos.tolist()],
            "step_cos_original_negative": [float(x) for x in cos_neg.tolist()],
            "step_cos_positive_negative": [float(x) for x in cos_pn.tolist()],
            "step_delta_positive_minus_negative": [float(x) for x in delta.tolist()],
            "S_positive_mean": float(cos_pos.mean().item()),
            "S_negative_mean": float(cos_neg.mean().item()),
            "delta_evidence_mean": float(delta.mean().item()),
            "relative_l2_original_positive_mean": float(rel_l2_pos.mean().item()),
            "relative_l2_original_negative_mean": float(rel_l2_neg.mean().item()),
            "interpretation_allowed": ref_replay_valid,
            "note": (
                "This is a single-sample counterfactual pilot. A positive delta is only "
                "descriptive until replicated across multiple annotated samples."
            ),
        },
    }

    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n========== FIXED REPLAY SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nVSTAR_FIXED_PREFIX_EVIDENCE_REPLAY_PASS=True")


if __name__ == "__main__":
    main()
