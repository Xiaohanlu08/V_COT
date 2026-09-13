#!/usr/bin/env python3
import json
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

LATENT_START_ID = 151666
LATENT_END_ID = 151667
NUM_CONTROLS = 32
CONTROL_SEED = 20260913
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


def boxes_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def expand_box(box, margin_x, margin_y, width, height):
    x0, y0, x1, y1 = box
    return [
        max(0, x0 - margin_x),
        max(0, y0 - margin_y),
        min(width, x1 + margin_x),
        min(height, y1 + margin_y),
    ]


def neutral_mask(image, box_xyxy):
    import numpy as np
    from PIL import Image

    arr = np.asarray(image.convert("RGB")).copy()
    h, w, _ = arr.shape
    x0, y0, x1, y1 = [int(v) for v in box_xyxy]
    bw = x1 - x0
    bh = y1 - y0
    pad = max(4, int(round(0.10 * max(bw, bh))))

    rx0 = max(0, x0 - pad)
    ry0 = max(0, y0 - pad)
    rx1 = min(w, x1 + pad)
    ry1 = min(h, y1 + pad)

    ring = np.zeros((h, w), dtype=bool)
    ring[ry0:ry1, rx0:rx1] = True
    ring[y0:y1, x0:x1] = False
    pixels = arr[ring]
    if pixels.size == 0:
        neutral = np.round(arr.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    else:
        neutral = np.round(pixels.mean(axis=0)).astype(np.uint8)

    arr[y0:y1, x0:x1] = neutral
    return Image.fromarray(arr, mode="RGB"), [int(x) for x in neutral.tolist()]


def sample_control_boxes(width, height, target_box, n=NUM_CONTROLS, seed=CONTROL_SEED):
    rng = random.Random(seed)
    x0, y0, x1, y1 = target_box
    bw = x1 - x0
    bh = y1 - y0

    exclusion = expand_box(target_box, bw, bh, width, height)

    controls = []
    attempts = 0
    while len(controls) < n and attempts < 100000:
        attempts += 1
        cx0 = rng.randint(0, width - bw)
        cy0 = rng.randint(0, height - bh)
        candidate = [cx0, cy0, cx0 + bw, cy0 + bh]
        if boxes_overlap(candidate, exclusion):
            continue
        if any(candidate == old for old in controls):
            continue
        controls.append(candidate)

    if len(controls) != n:
        raise RuntimeError(f"Could only sample {len(controls)} control boxes after {attempts} attempts")
    return controls, exclusion


def main():
    from importlib.metadata import version

    import numpy as np
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
    dump_dir = Path(os.environ["VCOT_LATENT_DUMP_DIR"]).resolve()
    result_dir = root / "results" / "occlusion_specificity" / "vstar_pos0"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("========== ENVIRONMENT ==========")
    print("VCOT_ROOT:", root)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", version("torch"))
    print("transformers:", version("transformers"))
    print("vllm:", version("vllm"))
    print("NUM_CONTROLS:", NUM_CONTROLS)
    print("CONTROL_SEED:", CONTROL_SEED)

    if Path(runner.__file__).name != "monet_gpu_model_runner.py":
        raise RuntimeError("Instrumented Monet runner is not active.")
    if os.environ.get("LATENT_SIZE") != "10":
        raise RuntimeError("Expected LATENT_SIZE=10")
    if os.environ.get("VCOT_LATENT_TENSOR_DUMP") != "1":
        raise RuntimeError("VCOT_LATENT_TENSOR_DUMP must equal 1")
    if os.environ.get("VCOT_FORCE_LATENT_START_ONCE") != "1":
        raise RuntimeError("VCOT_FORCE_LATENT_START_ONCE must equal 1")

    baseline_records = [
        json.loads(line)
        for line in baseline_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline = next(r for r in baseline_records if int(r["dataset_position"]) == 0)
    baseline_ids = list(baseline["token_ids"])
    baseline_segment = find_single_segment(baseline_ids)
    if baseline_segment != (25, 35):
        raise RuntimeError(f"Unexpected baseline segment: {baseline_segment}")
    prefix_ids = baseline_ids[:25]

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
    original = Image.open(original_path).convert("RGB")
    if original.size != (2000, 1500):
        raise RuntimeError(f"Unexpected original image size: {original.size}")

    target_xywh = [564, 142, 155, 157]
    tx, ty, tw, th = target_xywh
    target_box = [tx, ty, tx + tw, ty + th]

    controls, exclusion = sample_control_boxes(
        original.width, original.height, target_box, NUM_CONTROLS, CONTROL_SEED
    )

    target_mask_img, target_fill = neutral_mask(original, target_box)
    target_mask_path = result_dir / "I_target_mask_full.png"
    target_mask_img.save(target_mask_path)

    control_meta = []
    for i, box in enumerate(controls):
        _, fill = neutral_mask(original, box)
        control_meta.append({
            "index": i,
            "box_xyxy": box,
            "neutral_fill_rgb": fill,
        })

    metadata = {
        "dataset": "VStarBench",
        "dataset_position": 0,
        "target_object": "glove",
        "target_bbox_xywh": target_xywh,
        "target_box_xyxy": target_box,
        "target_neutral_fill_rgb": target_fill,
        "control_seed": CONTROL_SEED,
        "num_controls": NUM_CONTROLS,
        "control_exclusion_box_xyxy": exclusion,
        "controls": control_meta,
        "image_size": list(original.size),
    }
    (result_dir / "mask_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

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
            raise RuntimeError("Unexpected video input")
        if not images or len(images) != 1:
            raise RuntimeError("Expected exactly one processed image")

        base_ids = tokenizer.encode(text, add_special_tokens=False)
        return {
            "prompt_token_ids": list(base_ids) + list(prefix_ids),
            "multi_modal_data": {"image": images},
        }, base_ids, text

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        stop_token_ids=None,
    )

    def run_image(name, image_path, save_tensor=False):
        for p in dump_dir.glob("latent_step_*.pt"):
            p.unlink()

        payload, base_ids, template_text = prepare_payload(image_path)
        outputs = model.llm.generate(payload, sampling_params=sampling_params)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(f"Unexpected output shape for {name}")

        candidate = outputs[0].outputs[0]
        token_ids = list(candidate.token_ids)
        segment = find_single_segment(token_ids)
        files = sorted(dump_dir.glob("latent_step_*.pt"))
        tensors = [torch.load(p, map_location="cpu", weights_only=False) for p in files]
        if len(tensors) != 10:
            raise RuntimeError(f"Expected 10 latent tensors for {name}, got {len(tensors)}")
        z = torch.stack(tensors, dim=0).float()
        if tuple(z.shape) != (10, 3584) or not torch.isfinite(z).all().item():
            raise RuntimeError(f"Invalid latent tensor for {name}: {tuple(z.shape)}")
        if not token_ids or token_ids[0] != LATENT_START_ID or segment != (0, 10):
            raise RuntimeError(f"Mechanical alignment failed for {name}: {segment}")

        if save_tensor:
            torch.save(
                {
                    "view": name,
                    "image_path": str(image_path),
                    "tensor": z,
                    "token_ids_after_prefix": token_ids,
                },
                result_dir / f"{name}_latents.pt",
            )

        return {
            "z": z,
            "base_ids": base_ids,
            "template_text": template_text,
            "token_ids": token_ids,
            "raw_text": candidate.text,
        }

    print("\n========== ORIGINAL REPLAY ==========")
    original_run = run_image("original", original_path, save_tensor=True)
    z0 = original_run["z"]

    ref_cos = F.cosine_similarity(z_ref, z0, dim=1)
    ref_rel_l2 = (z_ref - z0).norm(dim=1) / z_ref.norm(dim=1).clamp_min(1e-12)
    replay_valid = bool(
        float(ref_cos.mean()) >= 0.999
        and float(ref_cos.min()) >= 0.995
        and float(ref_rel_l2.mean()) <= 0.05
    )
    print("original replay mean cosine vs EXP-0005:", float(ref_cos.mean()))
    print("original replay min cosine vs EXP-0005:", float(ref_cos.min()))
    print("original replay mean relative L2 vs EXP-0005:", float(ref_rel_l2.mean()))
    print("original replay valid:", replay_valid)
    if not replay_valid:
        raise RuntimeError("Original replay failed EXP-0005 reproduction gate.")

    print("\n========== TARGET MASK ==========")
    target_run = run_image("target_mask", target_mask_path, save_tensor=True)
    zt = target_run["z"]
    target_step_cos = F.cosine_similarity(z0, zt, dim=1)
    target_step_rel_l2 = (z0 - zt).norm(dim=1) / z0.norm(dim=1).clamp_min(1e-12)
    target_mean_cos = float(target_step_cos.mean())
    target_dist = 1.0 - target_mean_cos
    target_rel_l2 = float(target_step_rel_l2.mean())
    print("target mean cosine:", target_mean_cos)
    print("target cosine distance:", target_dist)
    print("target mean relative L2:", target_rel_l2)

    print("\n========== SHAM MASK CONTROLS ==========")
    control_results = []
    with tempfile.TemporaryDirectory(prefix="vcot_sham_masks_") as td:
        td = Path(td)
        for meta in control_meta:
            idx = meta["index"]
            box = meta["box_xyxy"]
            img, fill = neutral_mask(original, box)
            path = td / f"control_{idx:02d}.png"
            img.save(path)

            run = run_image(f"control_{idx:02d}", path, save_tensor=False)
            zc = run["z"]
            step_cos = F.cosine_similarity(z0, zc, dim=1)
            step_rel_l2 = (z0 - zc).norm(dim=1) / z0.norm(dim=1).clamp_min(1e-12)
            mean_cos = float(step_cos.mean())
            dist = 1.0 - mean_cos
            mean_rel_l2 = float(step_rel_l2.mean())

            row = {
                "index": idx,
                "box_xyxy": box,
                "neutral_fill_rgb": fill,
                "mean_cosine": mean_cos,
                "cosine_distance": dist,
                "mean_relative_l2": mean_rel_l2,
                "step_cosine": [float(x) for x in step_cos.tolist()],
            }
            control_results.append(row)
            print(
                f"[{idx+1:02d}/{NUM_CONTROLS}] "
                f"cos={mean_cos:.6f} dist={dist:.6f} rel_l2={mean_rel_l2:.6f}"
            )

    control_dists = np.asarray([r["cosine_distance"] for r in control_results], dtype=float)
    control_cos = np.asarray([r["mean_cosine"] for r in control_results], dtype=float)
    control_l2 = np.asarray([r["mean_relative_l2"] for r in control_results], dtype=float)

    num_controls_ge_target = int(np.sum(control_dists >= target_dist))
    empirical_p = (1 + num_controls_ge_target) / (NUM_CONTROLS + 1)
    percentile = 100.0 * float(np.sum(control_dists < target_dist)) / NUM_CONTROLS
    specificity_score = target_dist - float(np.median(control_dists))

    control_step_cos = np.asarray([r["step_cosine"] for r in control_results], dtype=float)
    median_control_step_cos = np.median(control_step_cos, axis=0)
    step_specificity = median_control_step_cos - target_step_cos.numpy()

    summary = {
        "dataset": "VStarBench",
        "dataset_position": 0,
        "protocol": "full-image target occlusion vs matched same-size sham occlusions under fixed prefix/fixed trigger",
        "num_controls": NUM_CONTROLS,
        "control_seed": CONTROL_SEED,
        "mechanical_alignment_pass": True,
        "original_replay_vs_EXP0005": {
            "mean_cosine": float(ref_cos.mean()),
            "min_cosine": float(ref_cos.min()),
            "mean_relative_l2": float(ref_rel_l2.mean()),
            "valid_for_counterfactual_comparison": replay_valid,
        },
        "target_mask": {
            "box_xyxy": target_box,
            "neutral_fill_rgb": target_fill,
            "mean_cosine": target_mean_cos,
            "cosine_distance": target_dist,
            "mean_relative_l2": target_rel_l2,
            "step_cosine": [float(x) for x in target_step_cos.tolist()],
        },
        "sham_controls": {
            "mean_cosine_mean": float(control_cos.mean()),
            "mean_cosine_median": float(np.median(control_cos)),
            "cosine_distance_mean": float(control_dists.mean()),
            "cosine_distance_median": float(np.median(control_dists)),
            "cosine_distance_std": float(control_dists.std(ddof=1)),
            "mean_relative_l2_mean": float(control_l2.mean()),
            "mean_relative_l2_median": float(np.median(control_l2)),
        },
        "target_specificity": {
            "cosine_distance_target_minus_control_median": specificity_score,
            "target_distance_percentile_among_controls": percentile,
            "num_controls_with_distance_ge_target": num_controls_ge_target,
            "empirical_one_sided_p": empirical_p,
            "step_specificity_control_median_cos_minus_target_cos": [
                float(x) for x in step_specificity.tolist()
            ],
            "interpretation": (
                "Positive specificity means masking the annotated target perturbs the latent "
                "trajectory more than a typical same-size sham mask. This is a single-sample pilot."
            ),
        },
        "controls": control_results,
    }

    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\n========== OCCLUSION SPECIFICITY SUMMARY ==========")
    print(json.dumps({
        "original_replay_valid": replay_valid,
        "target_mean_cosine": target_mean_cos,
        "target_cosine_distance": target_dist,
        "target_mean_relative_l2": target_rel_l2,
        "control_cosine_distance_median": float(np.median(control_dists)),
        "control_cosine_distance_mean": float(control_dists.mean()),
        "specificity_score": specificity_score,
        "target_distance_percentile_among_controls": percentile,
        "num_controls_with_distance_ge_target": num_controls_ge_target,
        "empirical_one_sided_p": empirical_p,
    }, indent=2))
    print("\nVSTAR_TARGET_OCCLUSION_SPECIFICITY_PASS=True")


if __name__ == "__main__":
    main()
