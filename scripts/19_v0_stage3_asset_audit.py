#!/usr/bin/env python3
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

EXPECTED_MONET_SHA = "08939998d3d643a73a316e349faa34f420429153"
DATA_SUBSETS = [
    "Visual_CoT",
    "CogCoM",
    "ReFocus",
    "Zebra_CoT_count",
    "Zebra_CoT_visual_search",
    "Zebra_CoT_geometry",
]


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_head(path: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def unique_existing(paths):
    out = []
    seen = set()
    for p in paths:
        p = Path(p).expanduser().resolve()
        if p.exists() and str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def bounded_find_dirs(roots, target_names, max_depth=5):
    hits = []
    seen = set()
    skip_names = {
        ".git", ".cache", "__pycache__", ".conda", "node_modules",
        "wandb", "logs", "results",
    }
    for root in roots:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            continue
        base_depth = len(root.parts)
        for cur, dirs, _files in os.walk(root):
            cur_path = Path(cur)
            depth = len(cur_path.parts) - base_depth
            dirs[:] = [d for d in dirs if d not in skip_names]
            if depth >= max_depth:
                dirs[:] = []
            for d in list(dirs):
                if d in target_names:
                    hit = (cur_path / d).resolve()
                    if str(hit) not in seen:
                        hits.append(hit)
                        seen.add(str(hit))
    return sorted(hits)


def disk_info(path: Path):
    total, used, free = shutil.disk_usage(path)
    gb = 1024 ** 3
    return {
        "path": str(path),
        "total_GiB": round(total / gb, 2),
        "used_GiB": round(used / gb, 2),
        "free_GiB": round(free / gb, 2),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    monet = root / "third_party" / "Monet"
    models_dir = root / "models"
    out_dir = root / "results" / "v0_prep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== V0 STAGE-3 ASSET AUDIT ==========")
    print("V_COT root:", root)
    print("Monet source:", monet)

    source_files = {
        "stage3_script": monet / "script_examples" / "sft_stage3.sh",
        "main": monet / "src" / "main.py",
        "trainer": monet / "src" / "trainer.py",
        "utils": monet / "src" / "utils.py",
        "modeling": monet / "monet_qwen_model" / "modeling_qwen2_5_vl_monet.py",
        "deepspeed_zero2": monet / "deepspeed" / "ds_zero2_gpu.json",
    }

    actual_sha = git_head(monet)
    source_report = {}
    for name, path in source_files.items():
        source_report[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    source_ready = (
        actual_sha == EXPECTED_MONET_SHA
        and all(x["exists"] for x in source_report.values())
    )

    packages = {
        name: pkg_version(name)
        for name in [
            "torch", "torchvision", "transformers", "trl", "accelerate",
            "datasets", "deepspeed", "qwen-vl-utils", "huggingface-hub",
        ]
    }

    direct_model_candidates = []
    if models_dir.is_dir():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir():
                direct_model_candidates.append(str(p.resolve()))

    search_roots = unique_existing([
        root,
        root.parent,
        Path.home() / "work",
        Path.home() / "datasets",
        Path.home() / "data",
        Path("/mnt/user6/work"),
        Path("/mnt/user6/datasets"),
        Path("/mnt/user6/data"),
    ])

    target_dir_names = {
        "Monet-SFT-125K",
        "Monet-SFT-7B",
        "Monet_checkpoints",
        "monet_precomputed_target_latent",
        "sft_stage1",
        "sft_stage2",
        "sft_stage3",
    }
    found_dirs = bounded_find_dirs(search_roots, target_dir_names, max_depth=5)

    dataset_roots = [p for p in found_dirs if p.name == "Monet-SFT-125K"]
    dataset_reports = []
    for ds_root in dataset_roots:
        subset_report = {}
        for subset in DATA_SUBSETS:
            train_json = ds_root / subset / "train.json"
            subset_report[subset] = {
                "path": str(train_json),
                "exists": train_json.is_file(),
                "size_bytes": train_json.stat().st_size if train_json.is_file() else None,
            }
        dataset_reports.append({
            "root": str(ds_root),
            "subsets": subset_report,
            "all_six_train_json_present": all(
                v["exists"] for v in subset_report.values()
            ),
        })

    monet7b = root / "models" / "Monet-7B"
    monet_sft_direct = root / "models" / "Monet-SFT-7B"
    sft_candidates = [
        str(p) for p in found_dirs
        if p.name == "Monet-SFT-7B"
    ]
    if monet_sft_direct.is_dir() and str(monet_sft_direct.resolve()) not in sft_candidates:
        sft_candidates.append(str(monet_sft_direct.resolve()))

    teacher_latent_candidates = [
        str(p) for p in found_dirs
        if p.name == "monet_precomputed_target_latent"
    ]
    staged_checkpoint_candidates = [
        str(p) for p in found_dirs
        if p.name in {"Monet_checkpoints", "sft_stage1", "sft_stage2", "sft_stage3"}
    ]

    dataset_ready = any(r["all_six_train_json_present"] for r in dataset_reports)
    sft_checkpoint_ready = len(sft_candidates) > 0
    final_monet_checkpoint_ready = (
        monet7b.is_dir() and (monet7b / "config.json").is_file()
    )

    hook_checks = {}
    if source_files["trainer"].is_file():
        trainer_text = source_files["trainer"].read_text(encoding="utf-8")
        hook_checks["CustomTrainerSFT_STAGE3"] = "class CustomTrainerSFT_STAGE3" in trainer_text
        hook_checks["stage3_latent_forward"] = "student_outputs_latent = model(**inputs)" in trainer_text
        hook_checks["stage3_ce_alignment_objective"] = (
            "loss = student_ce_loss + self.alignment_weight * alignment_loss" in trainer_text
        )
    if source_files["main"].is_file():
        main_text = source_files["main"].read_text(encoding="utf-8")
        hook_checks["collate_fn_sft_stage3"] = "def collate_fn_sft_stage3" in main_text
        hook_checks["student_pixel_values"] = "student_pixel_values" in main_text
        hook_checks["student_alignment_poss"] = "student_alignment_poss" in main_text

    summary = {
        "expected_monet_sha": EXPECTED_MONET_SHA,
        "actual_monet_sha": actual_sha,
        "source_ready": source_ready,
        "source_files": source_report,
        "hook_checks": hook_checks,
        "packages": packages,
        "search_roots": [str(p) for p in search_roots],
        "direct_model_dirs": direct_model_candidates,
        "found_training_asset_dirs": [str(p) for p in found_dirs],
        "dataset_reports": dataset_reports,
        "dataset_ready": dataset_ready,
        "final_monet7b_checkpoint_ready": final_monet_checkpoint_ready,
        "monet_sft7b_candidates": sft_candidates,
        "monet_sft7b_ready": sft_checkpoint_ready,
        "teacher_latent_candidates": teacher_latent_candidates,
        "staged_checkpoint_candidates": staged_checkpoint_candidates,
        "deepspeed_installed": packages["deepspeed"] is not None,
        "disk": [
            disk_info(root),
            disk_info(Path.home()),
        ],
        "interpretation": {
            "source_hook_audit_ready": source_ready and all(hook_checks.values()),
            "official_stage3_recipe_locally_complete": bool(
                source_ready
                and dataset_ready
                and teacher_latent_candidates
                and staged_checkpoint_candidates
                and packages["deepspeed"] is not None
            ),
            "continued_sft_v0_assets_minimum": bool(
                source_ready
                and dataset_ready
                and sft_checkpoint_ready
                and packages["deepspeed"] is not None
            ),
        },
    }

    out_path = out_dir / "stage3_asset_audit.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n========== SOURCE / HOOK CHECK ==========")
    print("actual_monet_sha:", actual_sha)
    print("source_ready:", source_ready)
    print("hook_checks:", json.dumps(hook_checks, indent=2))

    print("\n========== PACKAGE CHECK ==========")
    print(json.dumps(packages, indent=2))

    print("\n========== TRAINING ASSET DISCOVERY ==========")
    print("dataset_ready:", dataset_ready)
    print("monet_sft7b_ready:", sft_checkpoint_ready)
    print("final_monet7b_checkpoint_ready:", final_monet_checkpoint_ready)
    print("teacher_latent_candidates:")
    for p in teacher_latent_candidates:
        print("  ", p)
    print("staged_checkpoint_candidates:")
    for p in staged_checkpoint_candidates:
        print("  ", p)
    print("Monet-SFT-7B candidates:")
    for p in sft_candidates:
        print("  ", p)
    print("Monet-SFT-125K dataset reports:")
    for r in dataset_reports:
        print(json.dumps(r, indent=2, ensure_ascii=False))

    print("\n========== READINESS SUMMARY ==========")
    print(json.dumps(summary["interpretation"], indent=2))
    print("audit_json:", out_path)
    print("\nV0_STAGE3_ASSET_AUDIT_PASS=True")


if __name__ == "__main__":
    main()
