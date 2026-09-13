#!/usr/bin/env python3
import hashlib
import json
import random
from pathlib import Path

COHORT_SIZE = 12
SEED = 20260913
DEVELOPMENT_POSITION = 0


def main():
    root = Path(__file__).resolve().parents[1]
    mapping_path = (
        root / "results" / "annotation_audit" /
        "vstar_triggered72_annotation_mapping.jsonl"
    )
    out_dir = root / "results" / "replication_cohort"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)

    rows = [
        json.loads(line)
        for line in mapping_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    eligible = []
    exclusion_counts = {
        "unresolved": 0,
        "not_direct_attributes": 0,
        "not_single_target_single_bbox": 0,
        "development_position_0": 0,
        "invalid_image_or_bbox_metadata": 0,
        "invalid_latent_segment": 0,
    }

    for row in rows:
        pos = int(row["dataset_position"])
        if row.get("resolution") == "unresolved":
            exclusion_counts["unresolved"] += 1
            continue
        if row.get("category") != "direct_attributes":
            exclusion_counts["not_direct_attributes"] += 1
            continue
        if not row.get("single_target_single_bbox", False):
            exclusion_counts["not_single_target_single_bbox"] += 1
            continue
        if pos == DEVELOPMENT_POSITION:
            exclusion_counts["development_position_0"] += 1
            continue

        image_size = row.get("image_size")
        bbox_list = row.get("bbox")
        if (
            not isinstance(image_size, list) or len(image_size) != 2
            or not isinstance(bbox_list, list) or len(bbox_list) != 1
            or not isinstance(bbox_list[0], list) or len(bbox_list[0]) != 4
        ):
            exclusion_counts["invalid_image_or_bbox_metadata"] += 1
            continue

        w, h = [int(x) for x in image_size]
        x, y, bw, bh = [int(round(float(v))) for v in bbox_list[0]]
        if w <= 0 or h <= 0 or bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w or y + bh > h:
            exclusion_counts["invalid_image_or_bbox_metadata"] += 1
            continue

        s = row.get("baseline_latent_start_position")
        e = row.get("baseline_latent_end_position")
        if s is None or e is None or int(e) - int(s) != 10:
            exclusion_counts["invalid_latent_segment"] += 1
            continue

        eligible.append(row)

    eligible = sorted(eligible, key=lambda x: int(x["dataset_position"]))
    if len(eligible) < COHORT_SIZE:
        raise RuntimeError(
            f"Need at least {COHORT_SIZE} eligible direct-attribute samples, got {len(eligible)}"
        )

    rng = random.Random(SEED)
    selected = rng.sample(eligible, COHORT_SIZE)
    selected = sorted(selected, key=lambda x: int(x["dataset_position"]))

    cohort = []
    for row in selected:
        cohort.append({
            "dataset_position": int(row["dataset_position"]),
            "category": row["category"],
            "question": row["question"],
            "annotation_relpath": row["annotation_relpath"],
            "annotation_stem": row["annotation_stem"],
            "target_object": row["target_object"],
            "bbox": row["bbox"],
            "image_path": row["image_path"],
            "image_size": row["image_size"],
            "baseline_latent_start_position": int(row["baseline_latent_start_position"]),
            "baseline_latent_end_position": int(row["baseline_latent_end_position"]),
            "baseline_token_count": int(row["baseline_token_count"]),
            "mapping_resolution": row["resolution"],
            "option_compat_score": row.get("option_compat_score"),
        })

    canonical = json.dumps(
        cohort,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cohort_sha256 = hashlib.sha256(canonical).hexdigest()

    cohort_path = out_dir / "vstar_direct_attributes_confirmatory12.json"
    cohort_jsonl_path = out_dir / "vstar_direct_attributes_confirmatory12.jsonl"
    summary_path = out_dir / "vstar_direct_attributes_confirmatory12_summary.json"

    cohort_path.write_text(
        json.dumps(cohort, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    cohort_jsonl_path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cohort),
        encoding="utf-8",
    )

    summary = {
        "dataset": "VStarBench",
        "purpose": "confirmatory replication of full-image target-vs-sham latent occlusion specificity",
        "selection_is_outcome_blind": True,
        "selection_seed": SEED,
        "cohort_size": COHORT_SIZE,
        "development_sample_excluded": DEVELOPMENT_POSITION,
        "eligibility_rule": {
            "natural_latent_triggered": True,
            "category": "direct_attributes",
            "official_annotation_resolved": True,
            "single_target_single_bbox": True,
            "valid_image_bbox_metadata": True,
            "baseline_latent_segment_length": 10,
        },
        "reason_relative_position_excluded": (
            "relative-position questions require relational evidence and often multiple objects; "
            "they will be evaluated with a relation-aware intervention rather than mixed into "
            "this direct-attribute target-occlusion cohort"
        ),
        "mapped_rows_total": len(rows),
        "eligible_candidate_count": len(eligible),
        "exclusion_counts": exclusion_counts,
        "selected_positions": [x["dataset_position"] for x in cohort],
        "selected_annotation_relpaths": [x["annotation_relpath"] for x in cohort],
        "cohort_sha256": cohort_sha256,
        "cohort_json": str(cohort_path),
        "cohort_jsonl": str(cohort_jsonl_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("========== FROZEN REPLICATION COHORT ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n========== SELECTED SAMPLES ==========")
    for x in cohort:
        print(
            f"pos={x['dataset_position']:3d} "
            f"ann={x['annotation_relpath']:<40s} "
            f"target={x['target_object']} bbox={x['bbox']} "
            f"latent=[{x['baseline_latent_start_position']},{x['baseline_latent_end_position']}]"
        )

    print("\nVSTAR_CONFIRMATORY12_COHORT_FREEZE_PASS=True")


if __name__ == "__main__":
    main()
