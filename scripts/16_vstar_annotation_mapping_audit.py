#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def norm_text(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def local_options(row):
    out = []
    for key in ["A", "B", "C", "D"]:
        if key in row and str(row[key]).lower() != "nan":
            out.append(str(row[key]).strip())
    return out


def option_compat_score(local_opts, official_opts):
    if len(local_opts) != len(official_opts):
        return -1
    score = 0
    for lo, oo in zip(local_opts, official_opts):
        a = norm_text(lo)
        b = norm_text(oo)
        if a == b:
            score += 3
        elif a and b and (a in b or b in a):
            score += 2
        else:
            # weak last-token compatibility for forms such as
            # "rubber" vs "The material of the glove is rubber."
            at = a.split()
            bt = b.split()
            if at and bt and at[-1] == bt[-1]:
                score += 1
    return score


def main():
    from huggingface_hub import snapshot_download
    from PIL import Image
    from tqdm import tqdm
    from vlmeval.dataset import build_dataset

    root = Path(__file__).resolve().parents[1]
    baseline_path = root / "results" / "natural_trigger" / "vstar_n191_seed20260913.jsonl"
    out_dir = root / "results" / "annotation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    records = [
        json.loads(line)
        for line in baseline_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    triggered = sorted(
        [r for r in records if r.get("triggered")],
        key=lambda r: int(r["dataset_position"]),
    )
    if len(triggered) != 72:
        raise RuntimeError(f"Expected 72 naturally-triggered records, got {len(triggered)}")

    print("========== OFFICIAL V*BENCH ANNOTATION SNAPSHOT ==========")
    snapshot = Path(snapshot_download(
        repo_id="craigwu/vstar_bench",
        repo_type="dataset",
        allow_patterns=[
            "direct_attributes/*.json",
            "relative_position/*.json",
        ],
    )).resolve()
    print("snapshot:", snapshot)

    ann_files = sorted(
        list((snapshot / "direct_attributes").glob("*.json"))
        + list((snapshot / "relative_position").glob("*.json"))
    )
    print("official_annotation_files:", len(ann_files))
    if not ann_files:
        raise RuntimeError("No official V*Bench annotation JSON files found")

    official = []
    by_qcat = defaultdict(list)
    for path in ann_files:
        ann = json.loads(path.read_text(encoding="utf-8"))
        category = path.parent.name
        item = {
            "annotation_path": str(path),
            "annotation_relpath": str(path.relative_to(snapshot)),
            "annotation_stem": path.stem,
            "category": category,
            "question": ann.get("question"),
            "options": ann.get("options", []),
            "target_object": ann.get("target_object", []),
            "bbox": ann.get("bbox", []),
        }
        official.append(item)
        by_qcat[(category, norm_text(item["question"]))].append(item)

    dataset = build_dataset("VStarBench")
    print("dataset_rows:", len(dataset.data))
    print("dataset_columns:", list(dataset.data.columns))
    if len(dataset.data) != 191:
        raise RuntimeError(f"Expected 191 VStarBench rows, got {len(dataset.data)}")

    mapped = []
    unresolved = []

    print("\n========== MAPPING 72 NATURALLY-TRIGGERED SAMPLES ==========")
    for r in tqdm(triggered, desc="Mapping annotations", unit="sample"):
        pos = int(r["dataset_position"])
        category = str(r["category"])
        row = dataset.data.iloc[pos]
        question = str(row["question"]).strip()
        opts = local_options(row)

        candidates = list(by_qcat.get((category, norm_text(question)), []))
        resolution = None
        chosen = None
        scores = []

        if len(candidates) == 1:
            chosen = candidates[0]
            resolution = "unique_question_category"
        elif len(candidates) > 1:
            for c in candidates:
                score = option_compat_score(opts, c["options"])
                scores.append((score, c))
            scores.sort(key=lambda x: x[0], reverse=True)
            if len(scores) == 1 or scores[0][0] > scores[1][0]:
                chosen = scores[0][1]
                resolution = "question_category_plus_options"

        prompt = dataset.build_prompt(pos)
        image_items = [x for x in prompt if x.get("type") == "image"]
        image_path = None
        image_size = None
        if len(image_items) == 1:
            image_path = str(Path(image_items[0]["value"]).resolve())
            if Path(image_path).is_file():
                with Image.open(image_path) as im:
                    image_size = list(im.size)

        base = {
            "dataset_position": pos,
            "category": category,
            "question": question,
            "local_options": opts,
            "image_path": image_path,
            "image_size": image_size,
            "baseline_latent_start_position": r.get("start_positions", [None])[0]
                if r.get("start_positions") else None,
            "baseline_latent_end_position": r.get("end_positions", [None])[0]
                if r.get("end_positions") else None,
            "baseline_token_count": len(r.get("token_ids", [])),
            "candidate_count": len(candidates),
        }

        if chosen is None:
            base["resolution"] = "unresolved"
            base["candidate_relpaths"] = [c["annotation_relpath"] for c in candidates]
            base["candidate_scores"] = [
                {"score": s, "annotation_relpath": c["annotation_relpath"]}
                for s, c in scores
            ]
            unresolved.append(base)
            mapped.append(base)
            continue

        opt_score = option_compat_score(opts, chosen["options"])
        bbox_count = len(chosen["bbox"])
        target_count = len(chosen["target_object"])
        single_bbox = bbox_count == 1
        single_target = target_count == 1

        base.update({
            "resolution": resolution,
            "annotation_relpath": chosen["annotation_relpath"],
            "annotation_stem": chosen["annotation_stem"],
            "official_options": chosen["options"],
            "option_compat_score": opt_score,
            "target_object": chosen["target_object"],
            "bbox": chosen["bbox"],
            "target_count": target_count,
            "bbox_count": bbox_count,
            "single_target_single_bbox": single_target and single_bbox,
        })
        mapped.append(base)

    out_jsonl = out_dir / "vstar_triggered72_annotation_mapping.jsonl"
    out_jsonl.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in mapped),
        encoding="utf-8",
    )

    mapped_ok = [x for x in mapped if x.get("resolution") != "unresolved"]
    single_box = [x for x in mapped_ok if x.get("single_target_single_bbox")]
    cat_counts = Counter(x["category"] for x in mapped_ok)
    single_cat_counts = Counter(x["category"] for x in single_box)
    resolution_counts = Counter(x["resolution"] for x in mapped)

    summary = {
        "dataset": "VStarBench",
        "triggered_total": len(triggered),
        "official_annotation_file_count": len(ann_files),
        "mapped_count": len(mapped_ok),
        "unresolved_count": len(unresolved),
        "resolution_counts": dict(resolution_counts),
        "mapped_category_counts": dict(cat_counts),
        "single_target_single_bbox_count": len(single_box),
        "single_target_single_bbox_category_counts": dict(single_cat_counts),
        "unresolved_positions": [x["dataset_position"] for x in unresolved],
        "mapping_jsonl": str(out_jsonl),
    }
    summary_path = out_dir / "vstar_triggered72_annotation_mapping_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n========== ANNOTATION MAPPING SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n========== FIRST 12 MAPPED SAMPLES ==========")
    for x in mapped_ok[:12]:
        print(
            f"pos={x['dataset_position']:3d} cat={x['category']:<20s} "
            f"ann={x['annotation_relpath']:<40s} "
            f"targets={x['target_count']} bboxes={x['bbox_count']} "
            f"opt_score={x['option_compat_score']}"
        )

    if unresolved:
        print("\n========== UNRESOLVED ==========")
        for x in unresolved:
            print(
                f"pos={x['dataset_position']:3d} cat={x['category']} "
                f"candidates={x['candidate_count']} question={x['question']}"
            )

    print("\nVSTAR_TRIGGERED72_ANNOTATION_AUDIT_PASS=True")


if __name__ == "__main__":
    main()
