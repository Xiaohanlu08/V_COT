#!/usr/bin/env python3
import hashlib
import json
import math
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

LATENT_START_ID = 151666
LATENT_END_ID = 151667
NUM_CONTROLS = 32
CONTROL_SEED_BASE = 20260913
EXPECTED_COHORT_SHA256 = "f65ebdbefc8a73276877f4096d5cb8897339447c91a31dc68322a811dda487e4"
EXPECTED_POSITIONS = [6, 11, 22, 23, 27, 53, 57, 59, 67, 71, 102, 112]
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


def sample_control_boxes(width, height, target_box, n, seed):
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
        raise RuntimeError(
            f"Could only sample {len(controls)} controls after {attempts} attempts"
        )
    return controls, exclusion


def exact_one_sided_sign_test(positive_count, nonzero_count):
    if nonzero_count == 0:
        return 1.0
    tail = sum(
        math.comb(nonzero_count, k)
        for k in range(positive_count, nonzero_count + 1)
    )
    return tail / (2 ** nonzero_count)


def exact_signflip_mean_p(values):
    if not values:
        return 1.0
    magnitudes = [abs(float(v)) for v in values]
    observed = sum(float(v) for v in values) / len(values)
    n = len(values)
    ge = 0
    total = 1 << n
    for mask in range(total):
        s = 0.0
        for i, mag in enumerate(magnitudes):
            s += mag if ((mask >> i) & 1) else -mag
        perm_mean = s / n
        if perm_mean >= observed - 1e-15:
            ge += 1
    return ge / total


def canonical_sha256(obj):
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    from importlib.metadata import version

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from vllm import SamplingParams
    import vllm.v1.worker.gpu_model_runner as runner
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat

    root = Path(os.environ["VCOT_ROOT"]).resolve()
    model_dir = root / "models" / "Monet-7B"
    baseline_jsonl = root / "results" / "natural_trigger" / "vstar_n191_seed20260913.jsonl"
    cohort_path = root / "results" / "replication_cohort" / "vstar_direct_attributes_confirmatory12.json"
    dump_dir = Path(os.environ["VCOT_LATENT_DUMP_DIR"]).resolve()
    preforce_dir = Path(os.environ["VCOT_PREFORCE_DUMP_DIR"]).resolve()
    result_dir = root / "results" / "confirmatory_occlusion12"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("========== ENVIRONMENT ==========")
    print("VCOT_ROOT:", root)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", version("torch"))
    print("transformers:", version("transformers"))
    print("vllm:", version("vllm"))
    print("NUM_CONTROLS:", NUM_CONTROLS)
    print("CONTROL_SEED_BASE:", CONTROL_SEED_BASE)
    print("runner_file:", Path(runner.__file__).resolve())

    if Path(runner.__file__).name != "monet_gpu_model_runner.py":
        raise RuntimeError("Instrumented Monet runner is not active.")
    if os.environ.get("LATENT_SIZE") != "10":
        raise RuntimeError("Expected LATENT_SIZE=10")
    if os.environ.get("VCOT_LATENT_TENSOR_DUMP") != "1":
        raise RuntimeError("VCOT_LATENT_TENSOR_DUMP must equal 1")
    if os.environ.get("VCOT_FORCE_LATENT_START_ONCE") != "1":
        raise RuntimeError("VCOT_FORCE_LATENT_START_ONCE must equal 1")

    for required in [baseline_jsonl, cohort_path]:
        if not required.is_file():
            raise FileNotFoundError(required)

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_sha = canonical_sha256(cohort)
    positions = [int(x["dataset_position"]) for x in cohort]
    print("\n========== FROZEN COHORT CHECK ==========")
    print("cohort_sha256:", cohort_sha)
    print("positions:", positions)
    if cohort_sha != EXPECTED_COHORT_SHA256:
        raise RuntimeError(
            f"Frozen cohort SHA mismatch: expected {EXPECTED_COHORT_SHA256}, got {cohort_sha}"
        )
    if positions != EXPECTED_POSITIONS:
        raise RuntimeError(
            f"Frozen cohort positions changed: expected {EXPECTED_POSITIONS}, got {positions}"
        )

    baseline_records = [
        json.loads(line)
        for line in baseline_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_by_pos = {int(r["dataset_position"]): r for r in baseline_records}

    dataset = build_dataset("VStarBench")

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

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        stop_token_ids=None,
    )

    def prepare_payload(base_prompt, image_path, prefix_ids):
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
        payload = {
            "prompt_token_ids": list(base_ids) + list(prefix_ids),
            "multi_modal_data": {"image": images},
        }
        return payload, base_ids, text

    def run_image(name, base_prompt, image_path, prefix_ids, expected_base_ids=None, save_path=None):
        for p in dump_dir.glob("latent_step_*.pt"):
            p.unlink()
        preforce_file = preforce_dir / "preforce_token.txt"
        if preforce_file.exists():
            preforce_file.unlink()

        payload, base_ids, template_text = prepare_payload(
            base_prompt, image_path, prefix_ids
        )
        if expected_base_ids is not None and base_ids != expected_base_ids:
            raise RuntimeError(f"Base prompt token IDs changed for {name}")

        outputs = model.llm.generate(
            payload,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(f"Unexpected vLLM output shape for {name}")

        candidate = outputs[0].outputs[0]
        token_ids = list(candidate.token_ids)
        segment = find_single_segment(token_ids)

        if not preforce_file.is_file():
            raise RuntimeError(f"Missing pre-force token dump for {name}")
        pre_force_token = int(preforce_file.read_text(encoding="utf-8").strip())

        files = sorted(dump_dir.glob("latent_step_*.pt"))
        tensors = [
            torch.load(p, map_location="cpu", weights_only=False)
            for p in files
        ]
        if len(tensors) != 10:
            raise RuntimeError(
                f"Expected 10 latent tensors for {name}, found {len(tensors)}"
            )
        if any(t.ndim != 1 or int(t.numel()) != 3584 for t in tensors):
            raise RuntimeError(f"Unexpected latent shape for {name}")

        z = torch.stack(tensors, dim=0).float()
        if tuple(z.shape) != (10, 3584) or not torch.isfinite(z).all().item():
            raise RuntimeError(f"Invalid latent tensor for {name}: {tuple(z.shape)}")
        if not token_ids or token_ids[0] != LATENT_START_ID or segment != (0, 10):
            raise RuntimeError(
                f"Mechanical fixed-trigger alignment failed for {name}: segment={segment}"
            )

        if save_path is not None:
            torch.save(
                {
                    "name": name,
                    "image_path": str(image_path),
                    "pre_force_token": pre_force_token,
                    "token_ids_after_prefix": token_ids,
                    "tensor": z,
                },
                save_path,
            )

        return {
            "z": z,
            "base_ids": base_ids,
            "template_text": template_text,
            "pre_force_token": pre_force_token,
            "token_ids": token_ids,
            "raw_text": candidate.text,
        }

    all_results = []

    print("\n========== CONFIRMATORY 12-SAMPLE RUN ==========")
    for sample_index, item in enumerate(cohort, start=1):
        pos = int(item["dataset_position"])
        print(f"\n----- SAMPLE {sample_index:02d}/12 | position {pos} -----")

        if pos not in baseline_by_pos:
            raise RuntimeError(f"Missing baseline record for position {pos}")
        baseline = baseline_by_pos[pos]
        baseline_ids = list(baseline["token_ids"])
        start = int(item["baseline_latent_start_position"])
        end = int(item["baseline_latent_end_position"])
        if end - start != 10:
            raise RuntimeError(f"Unexpected latent length at position {pos}: {start},{end}")
        if baseline_ids[start] != LATENT_START_ID or baseline_ids[end] != LATENT_END_ID:
            raise RuntimeError(f"Baseline marker mismatch at position {pos}")
        prefix_ids = baseline_ids[:start]

        prefix_text = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prefix_roundtrip_exact = (
            tokenizer.encode(prefix_text, add_special_tokens=False) == prefix_ids
        )

        base_prompt = dataset.build_prompt(pos)
        image_items = [x for x in base_prompt if x.get("type") == "image"]
        if len(image_items) != 1:
            raise RuntimeError(f"Expected one image at position {pos}")
        dataset_image_path = Path(image_items[0]["value"]).resolve()
        frozen_image_path = Path(item["image_path"]).resolve()
        if dataset_image_path != frozen_image_path:
            raise RuntimeError(
                f"Image path mismatch at position {pos}: {dataset_image_path} vs {frozen_image_path}"
            )
        if not frozen_image_path.is_file():
            raise FileNotFoundError(frozen_image_path)

        original = Image.open(frozen_image_path).convert("RGB")
        frozen_size = tuple(int(x) for x in item["image_size"])
        if original.size != frozen_size:
            raise RuntimeError(
                f"Image size mismatch at position {pos}: {original.size} vs {frozen_size}"
            )

        bbox = item["bbox"][0]
        x, y, bw, bh = [int(round(float(v))) for v in bbox]
        target_box = [x, y, x + bw, y + bh]
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > original.width or y + bh > original.height:
            raise RuntimeError(f"Invalid target bbox at position {pos}: {bbox}")

        sample_dir = result_dir / f"pos_{pos:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"target={item['target_object']} bbox={bbox} image_size={original.size} "
            f"prefix_len={len(prefix_ids)}"
        )

        original_run = run_image(
            f"pos{pos}_original",
            base_prompt,
            frozen_image_path,
            prefix_ids,
            save_path=sample_dir / "original_latents.pt",
        )
        original_boundary_valid = original_run["pre_force_token"] == LATENT_START_ID
        print("original_pre_force_token:", original_run["pre_force_token"])
        print("original_boundary_valid:", original_boundary_valid)

        if not original_boundary_valid:
            invalid = {
                "dataset_position": pos,
                "mechanically_valid": False,
                "failure_reason": (
                    "original pre-force greedy token at frozen prefix was not latent-start"
                ),
                "original_pre_force_token": original_run["pre_force_token"],
                "prefix_roundtrip_exact": prefix_roundtrip_exact,
            }
            all_results.append(invalid)
            (sample_dir / "result.json").write_text(
                json.dumps(invalid, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("[INVALID] confirmatory sample retained but excluded from specificity calculation")
            continue

        z0 = original_run["z"]
        base_ids_ref = original_run["base_ids"]

        target_img, target_fill = neutral_mask(original, target_box)
        target_path = sample_dir / "I_target_mask_full.png"
        target_img.save(target_path)
        target_run = run_image(
            f"pos{pos}_target",
            base_prompt,
            target_path,
            prefix_ids,
            expected_base_ids=base_ids_ref,
            save_path=sample_dir / "target_latents.pt",
        )
        zt = target_run["z"]
        target_step_cos = F.cosine_similarity(z0, zt, dim=1)
        target_step_rel_l2 = (
            (z0 - zt).norm(dim=1) / z0.norm(dim=1).clamp_min(1e-12)
        )
        target_mean_cos = float(target_step_cos.mean().item())
        target_dist = 1.0 - target_mean_cos
        target_rel_l2 = float(target_step_rel_l2.mean().item())

        seed = CONTROL_SEED_BASE + pos
        controls, exclusion = sample_control_boxes(
            original.width,
            original.height,
            target_box,
            NUM_CONTROLS,
            seed,
        )

        control_results = []
        with tempfile.TemporaryDirectory(prefix=f"vcot_pos{pos}_sham_") as td:
            td = Path(td)
            for ci, box in enumerate(
                tqdm(controls, desc=f"pos {pos} sham controls", unit="mask")
            ):
                control_img, fill = neutral_mask(original, box)
                control_path = td / f"control_{ci:02d}.png"
                control_img.save(control_path)
                run = run_image(
                    f"pos{pos}_control_{ci:02d}",
                    base_prompt,
                    control_path,
                    prefix_ids,
                    expected_base_ids=base_ids_ref,
                )
                zc = run["z"]
                step_cos = F.cosine_similarity(z0, zc, dim=1)
                step_rel_l2 = (
                    (z0 - zc).norm(dim=1) / z0.norm(dim=1).clamp_min(1e-12)
                )
                mean_cos = float(step_cos.mean().item())
                dist = 1.0 - mean_cos
                control_results.append({
                    "index": ci,
                    "box_xyxy": box,
                    "neutral_fill_rgb": fill,
                    "pre_force_token": run["pre_force_token"],
                    "mean_cosine": mean_cos,
                    "cosine_distance": dist,
                    "mean_relative_l2": float(step_rel_l2.mean().item()),
                    "step_cosine": [float(v) for v in step_cos.tolist()],
                })

        control_dists = np.asarray(
            [r["cosine_distance"] for r in control_results], dtype=float
        )
        control_median = float(np.median(control_dists))
        control_mean = float(np.mean(control_dists))
        specificity = float(target_dist - control_median)
        num_ge = int(np.sum(control_dists >= target_dist))
        percentile = float(100.0 * np.mean(control_dists < target_dist))
        within_p = float((1 + num_ge) / (NUM_CONTROLS + 1))
        ratio = float(target_dist / control_median) if control_median > 0 else None

        result = {
            "dataset_position": pos,
            "mechanically_valid": True,
            "annotation_relpath": item["annotation_relpath"],
            "target_object": item["target_object"],
            "bbox_xywh": bbox,
            "target_box_xyxy": target_box,
            "image_size": list(original.size),
            "prefix_token_count": len(prefix_ids),
            "prefix_roundtrip_exact": prefix_roundtrip_exact,
            "original_pre_force_token": original_run["pre_force_token"],
            "target_pre_force_token": target_run["pre_force_token"],
            "target_neutral_fill_rgb": target_fill,
            "control_seed": seed,
            "control_exclusion_box_xyxy": exclusion,
            "num_controls": NUM_CONTROLS,
            "target_mean_cosine": target_mean_cos,
            "target_cosine_distance": target_dist,
            "target_mean_relative_l2": target_rel_l2,
            "target_step_cosine": [float(v) for v in target_step_cos.tolist()],
            "control_cosine_distance_median": control_median,
            "control_cosine_distance_mean": control_mean,
            "specificity_score": specificity,
            "target_to_median_control_distance_ratio": ratio,
            "target_distance_percentile_among_controls": percentile,
            "num_controls_with_distance_ge_target": num_ge,
            "empirical_one_sided_within_image_p": within_p,
            "controls": control_results,
        }
        all_results.append(result)
        (sample_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(
            f"RESULT pos={pos}: target_dist={target_dist:.9f} "
            f"control_median={control_median:.9f} specificity={specificity:+.9f} "
            f"percentile={percentile:.3f} within_p={within_p:.6f}"
        )

    per_sample_path = result_dir / "per_sample_results.jsonl"
    per_sample_path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_results),
        encoding="utf-8",
    )

    valid = [r for r in all_results if r.get("mechanically_valid")]
    invalid = [r for r in all_results if not r.get("mechanically_valid")]
    specs = [float(r["specificity_score"]) for r in valid]
    positives = sum(v > 0 for v in specs)
    negatives = sum(v < 0 for v in specs)
    zeros = sum(v == 0 for v in specs)
    nonzero = positives + negatives
    sign_p = exact_one_sided_sign_test(positives, nonzero)
    signflip_p = exact_signflip_mean_p(specs)

    median_spec = float(np.median(specs)) if specs else None
    mean_spec = float(np.mean(specs)) if specs else None
    mean_percentile = (
        float(np.mean([r["target_distance_percentile_among_controls"] for r in valid]))
        if valid else None
    )
    median_percentile = (
        float(np.median([r["target_distance_percentile_among_controls"] for r in valid]))
        if valid else None
    )
    all_mechanically_valid = len(valid) == len(cohort) == 12
    primary_supported = bool(
        all_mechanically_valid
        and median_spec is not None
        and median_spec > 0
        and sign_p < 0.05
    )

    summary = {
        "dataset": "VStarBench",
        "cohort_name": "direct_attributes_confirmatory12",
        "cohort_sha256": cohort_sha,
        "positions": positions,
        "num_controls_per_sample": NUM_CONTROLS,
        "control_seed_rule": "20260913 + dataset_position",
        "mechanically_valid_samples": len(valid),
        "mechanically_invalid_samples": len(invalid),
        "invalid_positions": [r["dataset_position"] for r in invalid],
        "all_12_mechanically_valid": all_mechanically_valid,
        "positive_specificity_count": positives,
        "negative_specificity_count": negatives,
        "zero_specificity_count": zeros,
        "nonzero_sign_test_n": nonzero,
        "specificity_mean": mean_spec,
        "specificity_median": median_spec,
        "target_percentile_mean": mean_percentile,
        "target_percentile_median": median_percentile,
        "exact_one_sided_sign_test_p": sign_p,
        "exact_signflip_mean_p": signflip_p,
        "primary_rule": (
            "all 12 mechanically valid AND median specificity > 0 "
            "AND exact one-sided sign-test p < 0.05"
        ),
        "primary_hypothesis_supported": primary_supported,
        "per_sample_compact": [
            {
                "dataset_position": r["dataset_position"],
                "mechanically_valid": r.get("mechanically_valid", False),
                "specificity_score": r.get("specificity_score"),
                "target_distance_percentile_among_controls": r.get(
                    "target_distance_percentile_among_controls"
                ),
                "empirical_one_sided_within_image_p": r.get(
                    "empirical_one_sided_within_image_p"
                ),
            }
            for r in all_results
        ],
        "protocol": "protocols/CONFIRMATORY_OCCLUSION_12.md",
        "per_sample_results_jsonl": str(per_sample_path),
    }

    summary_path = result_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n========== CONFIRMATORY 12-SAMPLE SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n========== PER-SAMPLE RESULTS ==========")
    for r in all_results:
        if not r.get("mechanically_valid"):
            print(
                f"pos={r['dataset_position']:3d} INVALID "
                f"preforce={r.get('original_pre_force_token')}"
            )
        else:
            print(
                f"pos={r['dataset_position']:3d} "
                f"spec={r['specificity_score']:+.9f} "
                f"pct={r['target_distance_percentile_among_controls']:6.3f} "
                f"within_p={r['empirical_one_sided_within_image_p']:.6f}"
            )

    print("\nVSTAR_CONFIRMATORY_OCCLUSION12_PASS=True")


if __name__ == "__main__":
    main()
