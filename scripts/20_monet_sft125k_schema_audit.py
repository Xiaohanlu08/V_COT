#!/usr/bin/env python3
import json
import os
from collections import Counter
from pathlib import Path

DATASET_REPO = "NOVAglow646/Monet-SFT-125K"
MODEL_REPO = "NOVAglow646/Monet-SFT-7B"
DATASET_REVISION = "c77a2df"
MODEL_REVISION = "main"
SUBSETS = [
    "Visual_CoT",
    "CogCoM",
    "ReFocus",
    "Zebra_CoT_count",
    "Zebra_CoT_visual_search",
    "Zebra_CoT_geometry",
]


def load_json_any(path: Path):
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise RuntimeError(f"Empty JSON file: {path}")
    if stripped[0] == "[":
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def content_images(message):
    out = []
    for item in message.get("content", []):
        if item.get("type") == "image" and item.get("image"):
            out.append(str(item["image"]))
    return out


def content_text(message):
    chunks = []
    for item in message.get("content", []):
        if item.get("type") == "text" and item.get("text") is not None:
            chunks.append(str(item["text"]))
    return "\n".join(chunks)


def truncate(text, n=700):
    text = text.replace("\r", " ").replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def main():
    from huggingface_hub import hf_hub_download

    root = Path(__file__).resolve().parents[1]
    meta_root = root / "data" / "Monet-SFT-125K-metadata"
    model_meta_root = root / "data" / "Monet-SFT-7B-metadata"
    result_dir = root / "results" / "v0_prep"
    meta_root.mkdir(parents=True, exist_ok=True)
    model_meta_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("========== METADATA-ONLY DOWNLOAD ==========")
    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT"))
    print("dataset_repo:", DATASET_REPO)
    print("dataset_revision:", DATASET_REVISION)
    print("model_repo:", MODEL_REPO)
    print("model_revision:", MODEL_REVISION)

    local_train_paths = {}
    for subset in SUBSETS:
        filename = f"{subset}/train.json"
        print(f"Downloading metadata file: {filename}")
        cached = Path(hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
        ))
        dst = meta_root / subset / "train.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != cached.stat().st_size:
            dst.write_bytes(cached.read_bytes())
        local_train_paths[subset] = dst

    model_files = [
        "stage3/config.json",
        "stage3/model.safetensors.index.json",
        "stage3/tokenizer_config.json",
        "stage3/preprocessor_config.json",
    ]
    model_meta_paths = {}
    for filename in model_files:
        print(f"Downloading model metadata file: {filename}")
        cached = Path(hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            repo_type="model",
            revision=MODEL_REVISION,
        ))
        dst = model_meta_root / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != cached.stat().st_size:
            dst.write_bytes(cached.read_bytes())
        model_meta_paths[filename] = dst

    index = json.loads(
        model_meta_paths["stage3/model.safetensors.index.json"].read_text(encoding="utf-8")
    )
    stage3_total_bytes = index.get("metadata", {}).get("total_size")

    print("\n========== DATASET SCHEMA AUDIT ==========")
    subset_reports = {}
    representative = []

    grand = Counter()
    assistant_image_count_hist = Counter()
    user_image_count_hist = Counter()

    for subset in SUBSETS:
        path = local_train_paths[subset]
        rows = load_json_any(path)

        stats = Counter()
        assistant_hist = Counter()
        user_hist = Counter()
        samples = []

        for idx, row in enumerate(rows):
            stats["rows"] += 1

            metadata = row.get("metadata", {})
            data = row.get("data", [])
            role_map = {}
            for msg in data:
                role_map.setdefault(msg.get("role"), []).append(msg)

            user_msgs = role_map.get("user", [])
            assistant_msgs = role_map.get("assistant", [])

            user_images = [
                p for msg in user_msgs for p in content_images(msg)
            ]
            assistant_images = [
                p for msg in assistant_msgs for p in content_images(msg)
            ]
            assistant_text = "\n".join(content_text(msg) for msg in assistant_msgs)

            ui = len(user_images)
            ai = len(assistant_images)
            user_hist[ui] += 1
            assistant_hist[ai] += 1

            if ui > 0:
                stats["rows_with_user_image"] += 1
            if ai > 0:
                stats["rows_with_assistant_image"] += 1
            if ai == 0:
                stats["rows_without_assistant_image"] += 1
            if "<observation>" in assistant_text and "</observation>" in assistant_text:
                stats["rows_with_observation_tags"] += 1
            if ui > 0 and ai > 0:
                stats["rows_with_user_and_assistant_images"] += 1

            if len(samples) < 3:
                samples.append({
                    "row_index": idx,
                    "metadata": metadata,
                    "roles": [m.get("role") for m in data],
                    "user_images": user_images,
                    "assistant_images": assistant_images,
                    "assistant_text_preview": truncate(assistant_text),
                })

        report = {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "rows": stats["rows"],
            "rows_with_user_image": stats["rows_with_user_image"],
            "rows_with_assistant_image": stats["rows_with_assistant_image"],
            "rows_without_assistant_image": stats["rows_without_assistant_image"],
            "rows_with_user_and_assistant_images": stats["rows_with_user_and_assistant_images"],
            "rows_with_observation_tags": stats["rows_with_observation_tags"],
            "assistant_image_count_hist": dict(sorted(assistant_hist.items())),
            "user_image_count_hist": dict(sorted(user_hist.items())),
            "representative_samples": samples,
        }
        subset_reports[subset] = report

        grand.update(stats)
        assistant_image_count_hist.update(assistant_hist)
        user_image_count_hist.update(user_hist)
        representative.extend(
            [{"subset": subset, **s} for s in samples[:1]]
        )

        print(f"\n--- {subset} ---")
        print(json.dumps({
            k: report[k] for k in [
                "rows",
                "rows_with_user_image",
                "rows_with_assistant_image",
                "rows_without_assistant_image",
                "rows_with_user_and_assistant_images",
                "rows_with_observation_tags",
                "assistant_image_count_hist",
            ]
        }, indent=2, ensure_ascii=False))

    total_rows = grand["rows"]
    assistant_fraction = (
        grand["rows_with_assistant_image"] / total_rows if total_rows else 0.0
    )
    observation_fraction = (
        grand["rows_with_observation_tags"] / total_rows if total_rows else 0.0
    )

    summary = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision_requested": DATASET_REVISION,
        "model_repo": MODEL_REPO,
        "model_revision_requested": MODEL_REVISION,
        "metadata_only": True,
        "full_image_archives_downloaded": False,
        "stage3_weight_shards_downloaded": False,
        "stage3_total_weight_bytes_from_index": stage3_total_bytes,
        "total_rows": total_rows,
        "rows_with_user_image": grand["rows_with_user_image"],
        "rows_with_assistant_image": grand["rows_with_assistant_image"],
        "rows_with_user_and_assistant_images": grand["rows_with_user_and_assistant_images"],
        "rows_with_observation_tags": grand["rows_with_observation_tags"],
        "assistant_image_fraction": assistant_fraction,
        "observation_tag_fraction": observation_fraction,
        "assistant_image_count_hist": dict(sorted(assistant_image_count_hist.items())),
        "user_image_count_hist": dict(sorted(user_image_count_hist.items())),
        "subset_reports": subset_reports,
        "representative_one_per_subset": representative,
    }

    out = result_dir / "monet_sft125k_schema_audit.json"
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n========== GLOBAL SCHEMA SUMMARY ==========")
    print(json.dumps({
        "total_rows": summary["total_rows"],
        "rows_with_user_image": summary["rows_with_user_image"],
        "rows_with_assistant_image": summary["rows_with_assistant_image"],
        "rows_with_user_and_assistant_images": summary["rows_with_user_and_assistant_images"],
        "rows_with_observation_tags": summary["rows_with_observation_tags"],
        "assistant_image_fraction": summary["assistant_image_fraction"],
        "observation_tag_fraction": summary["observation_tag_fraction"],
        "assistant_image_count_hist": summary["assistant_image_count_hist"],
        "stage3_total_weight_bytes_from_index": summary["stage3_total_weight_bytes_from_index"],
    }, indent=2, ensure_ascii=False))

    print("\n========== REPRESENTATIVE ONE PER SUBSET ==========")
    for s in representative:
        print("\n" + "=" * 100)
        print("subset:", s["subset"])
        print("metadata:", json.dumps(s["metadata"], ensure_ascii=False))
        print("user_images:", s["user_images"])
        print("assistant_images:", s["assistant_images"])
        print("assistant_text_preview:", s["assistant_text_preview"])

    print("\naudit_json:", out)
    print("MONET_SFT125K_SCHEMA_AUDIT_PASS=True")


if __name__ == "__main__":
    main()
