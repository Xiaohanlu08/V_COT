#!/usr/bin/env python3
import json
import random
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageOps

DATASET_REPO = "NOVAglow646/Monet-SFT-125K"
DATASET_REVISION = "c77a2df"
SEED = 20260913
N = 12
BASES = [
    "https://hf-mirror.com",
    "https://hf-mirror.net",
    "https://huggingface.co",
]


def load_json_any(path: Path):
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise RuntimeError(f"Empty JSON file: {path}")
    if stripped[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def content_images(message):
    return [
        str(item["image"])
        for item in message.get("content", [])
        if item.get("type") == "image" and item.get("image")
    ]


def content_text(message):
    return "\n".join(
        str(item["text"])
        for item in message.get("content", [])
        if item.get("type") == "text" and item.get("text") is not None
    )


def extract_row(row):
    role_map = {}
    for msg in row.get("data", []):
        role_map.setdefault(msg.get("role"), []).append(msg)
    user_msgs = role_map.get("user", [])
    assistant_msgs = role_map.get("assistant", [])
    user_images = [p for m in user_msgs for p in content_images(m)]
    assistant_images = [p for m in assistant_msgs for p in content_images(m)]
    assistant_text = "\n".join(content_text(m) for m in assistant_msgs)
    user_text = "\n".join(content_text(m) for m in user_msgs)
    return user_images, assistant_images, user_text, assistant_text


def direct_download(filename: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        print(f"[reuse] {dst}")
        return
    if shutil.which("curl") is None:
        raise RuntimeError("curl not found; do not install anything yet")

    encoded = "/".join(quote(x) for x in filename.split("/"))
    errors = []
    for base in BASES:
        url = (
            f"{base}/datasets/{DATASET_REPO}/resolve/"
            f"{DATASET_REVISION}/{encoded}?download=true"
        )
        tmp = dst.with_suffix(dst.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        print(f"[download] {base}: {filename}")
        proc = subprocess.run(
            [
                "curl", "-fL", "--retry", "4", "--retry-delay", "2",
                "--connect-timeout", "20", "--max-time", "900",
                "--progress-bar", "-o", str(tmp), url,
            ],
            check=False,
        )
        if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(dst)
            return
        if tmp.exists():
            tmp.unlink()
        errors.append(f"{base}: rc={proc.returncode}")
    raise RuntimeError(f"Could not download {filename}: " + " | ".join(errors))


def fit_panel(img: Image.Image, size=(620, 390)):
    canvas = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(img.convert("RGB"), size)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def make_sheet(samples, out_path: Path, title: str):
    panel_w, panel_h = 620, 390
    label_h = 52
    margin = 18
    row_h = panel_h + label_h + margin
    width = margin + panel_w * 2 + margin * 2
    height = 74 + len(samples) * row_h + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 20), title, fill="black")
    draw.text((margin, 44), "left = original user image | right = assistant/helper image", fill="black")

    y = 74
    for s in samples:
        u = Image.open(s["user_local"]).convert("RGB")
        a = Image.open(s["assistant_local"]).convert("RGB")
        sheet.paste(fit_panel(u, (panel_w, panel_h)), (margin, y))
        sheet.paste(fit_panel(a, (panel_w, panel_h)), (margin * 2 + panel_w, y))
        label = (
            f"row={s['row_index']} sample_id={s['sample_id']}  "
            f"orig={s['user_size'][0]}x{s['user_size'][1]}  "
            f"helper={s['assistant_size'][0]}x{s['assistant_size'][1]}"
        )
        draw.text((margin, y + panel_h + 8), label, fill="black")
        y += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main():
    root = Path(__file__).resolve().parents[1]
    train_json = root / "data" / "Monet-SFT-125K-metadata" / "Visual_CoT" / "train.json"
    if not train_json.is_file():
        raise FileNotFoundError(
            f"Missing {train_json}; run Step 20b successfully first."
        )

    rows = load_json_any(train_json)
    eligible = []
    for idx, row in enumerate(rows):
        user_images, assistant_images, user_text, assistant_text = extract_row(row)
        if len(user_images) != 1 or len(assistant_images) != 1:
            continue
        if "<observation>" not in assistant_text or "</observation>" not in assistant_text:
            continue
        eligible.append({
            "row_index": idx,
            "sample_id": row.get("metadata", {}).get("sample_id"),
            "user_image": user_images[0],
            "assistant_image": assistant_images[0],
            "user_text": user_text,
            "assistant_text": assistant_text,
        })

    if len(eligible) < N:
        raise RuntimeError(f"Only {len(eligible)} eligible rows")

    rng = random.Random(SEED)
    selected = rng.sample(eligible, N)
    selected = sorted(selected, key=lambda x: x["row_index"])

    image_root = root / "data" / "Monet-SFT-125K-audit"
    out_dir = root / "results" / "v0_prep" / "visual_cot_helper_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    print("========== VISUAL_COT HELPER IMAGE AUDIT ==========")
    print("eligible_rows:", len(eligible))
    print("selection_seed:", SEED)
    print("selected_rows:", [x["row_index"] for x in selected])

    for i, s in enumerate(selected):
        user_dst = image_root / s["user_image"]
        assistant_dst = image_root / s["assistant_image"]
        direct_download(s["user_image"], user_dst)
        direct_download(s["assistant_image"], assistant_dst)

        with Image.open(user_dst) as im:
            user_size = list(im.size)
        with Image.open(assistant_dst) as im:
            assistant_size = list(im.size)

        area_ratio = (
            assistant_size[0] * assistant_size[1]
            / max(1, user_size[0] * user_size[1])
        )

        rec = {
            **s,
            "user_local": str(user_dst),
            "assistant_local": str(assistant_dst),
            "user_size": user_size,
            "assistant_size": assistant_size,
            "helper_to_original_pixel_area_ratio": area_ratio,
        }
        manifest.append(rec)

        print(
            f"row={s['row_index']:6d} sample_id={s['sample_id']} "
            f"orig={user_size} helper={assistant_size} area_ratio={area_ratio:.4f}"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sheets = []
    for k in range(0, N, 4):
        sheet_path = out_dir / f"contact_sheet_{k//4 + 1:02d}.png"
        make_sheet(
            manifest[k:k+4],
            sheet_path,
            f"Visual_CoT helper audit — samples {k+1}-{min(k+4, N)} of {N}",
        )
        sheets.append(str(sheet_path))

    print("\n========== AUDIT OUTPUTS ==========")
    print("manifest:", manifest_path)
    for p in sheets:
        print("contact_sheet:", p)

    print("\n========== TEXT PREVIEWS ==========")
    for r in manifest:
        preview = r["assistant_text"].replace("\n", " ")[:500]
        print(f"row={r['row_index']:6d} :: {preview}")

    print("\nVISUAL_COT_HELPER_AUDIT_PASS=True")


if __name__ == "__main__":
    main()
