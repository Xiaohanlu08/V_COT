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


def vstar_patch_box(bbox, image_width, image_height, patch_scale=1.2):
    """Replicate the official V*Bench target-patch geometry."""
    import math

    object_width = int(math.ceil(bbox[2]))
    object_height = int(math.ceil(bbox[3]))
    object_center_x = int(bbox[0] + bbox[2] / 2)
    object_center_y = int(bbox[1] + bbox[3] / 2)

    patch_width = int(object_width * patch_scale)
    patch_height = int(object_height * patch_scale)

    left = max(0, object_center_x - patch_width // 2)
    right = min(left + patch_width, image_width)
    top = max(0, object_center_y - patch_height // 2)
    bottom = min(top + patch_height, image_height)
    return [left, top, right, bottom]


def build_views(original, bbox, patch_scale=1.2):
    import numpy as np
    from PIL import Image

    crop_box = vstar_patch_box(bbox, original.width, original.height, patch_scale)
    left, top, right, bottom = crop_box

    positive_crop = original.crop((left, top, right, bottom)).convert("RGB")

    # Convert the target box into crop-local coordinates.
    bx, by, bw, bh = [int(round(x)) for x in bbox]
    lx0 = max(0, bx - left)
    ly0 = max(0, by - top)
    lx1 = min(positive_crop.width, bx + bw - left)
    ly1 = min(positive_crop.height, by + bh - top)
    local_bbox = [lx0, ly0, lx1, ly1]

    arr = np.asarray(positive_crop).copy()
    h, w, _ = arr.shape

    # Estimate a neutral fill from the surrounding ring, excluding the target.
    pad = max(4, int(round(0.10 * max(bw, bh))))
    rx0 = max(0, lx0 - pad)
    ry0 = max(0, ly0 - pad)
    rx1 = min(w, lx1 + pad)
    ry1 = min(h, ly1 + pad)

    ring_mask = np.zeros((h, w), dtype=bool)
    ring_mask[ry0:ry1, rx0:rx1] = True
    ring_mask[ly0:ly1, lx0:lx1] = False
    ring_pixels = arr[ring_mask]
    if ring_pixels.size == 0:
        neutral_rgb = np.round(arr.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    else:
        neutral_rgb = np.round(ring_pixels.mean(axis=0)).astype(np.uint8)

    negative_arr = arr.copy()
    negative_arr[ly0:ly1, lx0:lx1] = neutral_rgb
    negative_crop = Image.fromarray(negative_arr, mode="RGB")

    # Keep image dimensions identical to the original input so Qwen2.5-VL sees
    # the same nominal image size in all three conditions. Positive and negative
    # views are the same crop/resizing transform; they differ only in target evidence.
    positive = positive_crop.resize(original.size, Image.Resampling.LANCZOS)
    negative = negative_crop.resize(original.size, Image.Resampling.LANCZOS)

    return positive, negative, {
        "patch_scale": patch_scale,
        "crop_box_xyxy": crop_box,
        "crop_size_before_resize": [positive_crop.width, positive_crop.height],
        "local_target_bbox_xyxy": local_bbox,
        "neutral_fill_rgb": [int(x) for x in neutral_rgb.tolist()],
        "output_size": [original.width, original.height],
    }


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
    from huggingface_hub import hf_hub_download
    import vllm.v1.worker.gpu_model_runner as runner
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    root = Path(os.environ["VCOT_ROOT"]).resolve()
    model_dir = root / "models" / "Monet-7B"
    baseline_jsonl = root / "results" / "natural_trigger" / "vstar_n191_seed20260913.jsonl"
    dump_dir = Path(os.environ["VCOT_LATENT_DUMP_DIR"]).resolve()
    result_dir = root / "results" / "evidence_pilot" / "vstar_pos0"
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

    baseline_records = [
        json.loads(line)
        for line in baseline_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = next(r for r in baseline_records if int(r["dataset_position"]) == 0)
    if not baseline.get("triggered"):
        raise RuntimeError("Expected VStarBench position 0 to be naturally triggered.")

    print("\n========== OFFICIAL VSTAR ANNOTATION ==========")
    ann_path = hf_hub_download(
        repo_id="craigwu/vstar_bench",
        filename="direct_attributes/sa_4690.json",
        repo_type="dataset",
        local_files_only=True,
    )
    ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    print(json.dumps(ann, indent=2, ensure_ascii=False))

    if ann.get("target_object") != ["glove"]:
        raise RuntimeError(f"Unexpected target_object: {ann.get('target_object')}")
    if ann.get("bbox") != [[564, 142, 155, 157]]:
        raise RuntimeError(f"Unexpected bbox: {ann.get('bbox')}")
    if ann.get("question") != "What is the material of the glove?":
        raise RuntimeError("Official annotation question mismatch.")

    dataset = build_dataset("VStarBench")
    base_prompt = dataset.build_prompt(0)
    image_items = [x for x in base_prompt if x.get("type") == "image"]
    if len(image_items) != 1:
        raise RuntimeError(f"Expected one image item, got {len(image_items)}")
    original_path = Path(image_items[0]["value"]).resolve()
    original = Image.open(original_path).convert("RGB")
    if original.size != (2000, 1500):
        raise RuntimeError(f"Unexpected local image size: {original.size}")

    positive, negative, view_meta = build_views(original, ann["bbox"][0], patch_scale=1.2)
    positive_path = result_dir / "I_positive_zoom.png"
    negative_path = result_dir / "I_negative_mask.png"
    positive.save(positive_path)
    negative.save(negative_path)

    print("\n========== VIEW CONSTRUCTION ==========")
    print("original_path:", original_path)
    print("positive_path:", positive_path)
    print("negative_path:", negative_path)
    print(json.dumps(view_meta, indent=2))

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

    capture = {}
    original_generate = model.llm.generate

    def capture_generate(*args, **kwargs):
        outputs = original_generate(*args, **kwargs)
        capture["outputs"] = outputs
        capture["sampling_params"] = kwargs.get("sampling_params")
        return outputs

    model.llm.generate = capture_generate

    def run_view(name, image_path):
        for p in dump_dir.glob("latent_step_*.pt"):
            p.unlink()
        capture.clear()

        prompt = deepcopy(base_prompt)
        for item in prompt:
            if item.get("type") == "image":
                item["value"] = str(image_path)

        returned_text = model.generate(prompt, dataset="VStarBench")
        outputs = capture.get("outputs")
        if outputs is None or len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(f"Unexpected output shape for {name}")
        candidate = outputs[0].outputs[0]
        token_ids = list(candidate.token_ids)
        raw_text = candidate.text
        if returned_text != raw_text:
            raise RuntimeError(f"Returned/raw text mismatch for {name}")

        segment = find_single_segment(token_ids)
        files = sorted(dump_dir.glob("latent_step_*.pt"))
        z = None
        if files:
            tensors = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
            if any(t.ndim != 1 for t in tensors):
                raise RuntimeError(f"Non-1D latent tensor for {name}")
            z = torch.stack(tensors, dim=0).float()
            if not torch.isfinite(z).all().item():
                raise RuntimeError(f"Non-finite latent tensor for {name}")
            torch.save(
                {
                    "view": name,
                    "image_path": str(image_path),
                    "tensor": z,
                    "token_ids": token_ids,
                    "raw_text": raw_text,
                },
                result_dir / f"{name}_latents.pt",
            )

        return {
            "name": name,
            "image_path": str(image_path),
            "token_ids": token_ids,
            "raw_text": raw_text,
            "segment": segment,
            "num_latent_tensors": 0 if z is None else int(z.shape[0]),
            "hidden_size": None if z is None else int(z.shape[1]),
            "tensor": z,
        }

    print("\n========== NATURAL THREE-VIEW RUN ==========")
    runs = {
        "original": run_view("original", original_path),
        "positive": run_view("positive", positive_path),
        "negative": run_view("negative", negative_path),
    }

    original_exact = runs["original"]["token_ids"] == list(baseline["token_ids"])
    if not original_exact:
        raise RuntimeError("Original view no longer matches the recorded Step-09 baseline.")

    for name, r in runs.items():
        print(
            f"{name}: segment={r['segment']} latent_tensors={r['num_latent_tensors']} "
            f"generated_tokens={len(r['token_ids'])}"
        )

    all_trigger = all(r["segment"] is not None for r in runs.values())
    all_ten = all(r["num_latent_tensors"] == 10 for r in runs.values())

    prefix_match_pos = False
    prefix_match_neg = False
    same_trigger_position = False
    if all_trigger:
        so, _ = runs["original"]["segment"]
        sp, _ = runs["positive"]["segment"]
        sn, _ = runs["negative"]["segment"]
        prefix_o = runs["original"]["token_ids"][:so]
        prefix_p = runs["positive"]["token_ids"][:sp]
        prefix_n = runs["negative"]["token_ids"][:sn]
        prefix_match_pos = prefix_o == prefix_p
        prefix_match_neg = prefix_o == prefix_n
        same_trigger_position = so == sp == sn

    aligned_ready = bool(
        all_trigger
        and all_ten
        and prefix_match_pos
        and prefix_match_neg
        and same_trigger_position
    )

    summary = {
        "dataset": "VStarBench",
        "dataset_position": 0,
        "target_object": ann["target_object"],
        "bbox_xywh": ann["bbox"][0],
        "question": ann["question"],
        "view_construction": view_meta,
        "original_generation_exact_match_baseline": original_exact,
        "views": {
            name: {
                "segment": None if r["segment"] is None else list(r["segment"]),
                "num_latent_tensors": r["num_latent_tensors"],
                "hidden_size": r["hidden_size"],
                "generated_tokens": len(r["token_ids"]),
                "raw_text": r["raw_text"],
            }
            for name, r in runs.items()
        },
        "all_three_naturally_triggered": all_trigger,
        "all_three_have_10_latents": all_ten,
        "same_pre_latent_prefix_original_vs_positive": prefix_match_pos,
        "same_pre_latent_prefix_original_vs_negative": prefix_match_neg,
        "same_latent_start_position": same_trigger_position,
        "aligned_natural_comparison_ready": aligned_ready,
    }

    if all_ten:
        z0 = runs["original"]["tensor"]
        zp = runs["positive"]["tensor"]
        zn = runs["negative"]["tensor"]
        cos_pos = F.cosine_similarity(z0, zp, dim=1)
        cos_neg = F.cosine_similarity(z0, zn, dim=1)
        cos_pn = F.cosine_similarity(zp, zn, dim=1)
        delta = cos_pos - cos_neg
        summary["exploratory_similarity"] = {
            "step_cos_original_positive": [float(x) for x in cos_pos.tolist()],
            "step_cos_original_negative": [float(x) for x in cos_neg.tolist()],
            "step_cos_positive_negative": [float(x) for x in cos_pn.tolist()],
            "step_delta_positive_minus_negative": [float(x) for x in delta.tolist()],
            "S_positive_mean": float(cos_pos.mean().item()),
            "S_negative_mean": float(cos_neg.mean().item()),
            "delta_evidence_mean": float(delta.mean().item()),
            "interpretation_allowed": aligned_ready,
            "note": (
                "If aligned_natural_comparison_ready is false, these similarities are "
                "exploratory only because the textual pre-latent trajectory differs."
            ),
        }

    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n========== EVIDENCE PILOT SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nVSTAR_EVIDENCE_PILOT_CAPTURE_PASS=True")


if __name__ == "__main__":
    main()
