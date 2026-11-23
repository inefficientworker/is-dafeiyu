"""Fetch and curate positive (Dafeiyu) and in-repo negative images from EDMOK/blue-fish-archive."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

REPO_URL = "https://github.com/EDMOK/blue-fish-archive.git"
RAW_REPO_DIR = Path("data/raw/blue-fish-archive")
OUT_POS_DIR = Path("data/processed/positive")
OUT_NEG_AI_DIR = Path("data/processed/negative_ai_mascots")
META_PATH = Path("data/processed/repo_meta.jsonl")

POSITIVE_KEYWORDS = (
    "deepseek",
    "大肥鱼",
    "肥鱼",
    "鲸鱼",
    "蓝鱼",
    "小蓝鱼",
    "fish",
    "whale",
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
REVIEWED_POSITIVES = (
    "logo/deepseek_蓝鲸_彩色.png",
    "logo/deepseek_蓝鲸_黑色.png",
    "原始立绘/deepseek_whale.png",
    "art/v1/deepseek-chibi-hug.webp",
    "memes/120302wg44ju245iDQ38xu.png",
)


def clone_or_update_repo() -> None:
    if (RAW_REPO_DIR / ".git").exists():
        print(f"[repo] Already cloned at {RAW_REPO_DIR}")
        return
    RAW_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[repo] Cloning {REPO_URL} (--depth=1) ...")
    subprocess.run(
        ["git", "clone", "--depth=1", REPO_URL, str(RAW_REPO_DIR)],
        check=True,
    )


def extract_frames(src_path: Path, dst_dir: Path, stem_prefix: str, max_frames: int = 3) -> list[Path]:
    """Extract 1 static image or up to max_frames distinct keyframes from animated images."""
    saved: list[Path] = []
    with Image.open(src_path) as img:
        n_frames = getattr(img, "n_frames", 1)
        if n_frames <= 1:
            frame = ImageOps.exif_transpose(img)
            if frame.mode not in ("RGB", "RGBA"):
                frame = frame.convert("RGBA" if "A" in frame.getbands() else "RGB")
            out_path = dst_dir / f"{stem_prefix}.png"
            frame.save(out_path, format="PNG")
            saved.append(out_path)
            return saved

        if n_frames <= max_frames:
            indices = list(range(n_frames))
        else:
            indices = sorted({0, n_frames // 2, n_frames - 1})

        kept_hashes: list[imagehash.ImageHash] = []
        for idx in indices:
            img.seek(idx)
            frame = img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" else "RGB")
            h = imagehash.dhash(frame.convert("RGB"))
            if any((h - prev) < 4 for prev in kept_hashes):
                continue
            kept_hashes.append(h)
            out_path = dst_dir / f"{stem_prefix}_f{idx:03d}.png"
            frame.save(out_path, format="PNG")
            saved.append(out_path)
    return saved


def is_positive_text(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in POSITIVE_KEYWORDS)


def mascot_subcategory(src_path: Path, fallback: str) -> str:
    return "negative_gemini" if src_path.stem.lower().startswith("gemini-") else fallback


def main() -> None:
    clone_or_update_repo()
    OUT_POS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NEG_AI_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # 1. Process stickers/manifest.json (original files in media/ only — skip previews/ and large/)
    manifest_path = RAW_REPO_DIR / "stickers" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        rel_orig = entry["original"]
        src_path = RAW_REPO_DIR / rel_orig
        if not src_path.exists():
            continue
        text = f"{entry.get('filename', '')} {entry.get('alt', '')}"
        is_pos = is_positive_text(text)
        dst_dir = OUT_POS_DIR if is_pos else OUT_NEG_AI_DIR
        group_id = f"repo_media_{src_path.stem}"
        try:
            saved_paths = extract_frames(src_path, dst_dir, group_id, max_frames=3)
        except Exception as exc:
            print(f"[warn] Failed to decode {src_path}: {exc}")
            continue
        for p in saved_paths:
            records.append(
                {
                    "path": str(p),
                    "label": 1 if is_pos else 0,
                    "category": "positive_dafeiyu" if is_pos else "negative_blue_chars",
                    "subcategory": "repo_sticker" if is_pos else mascot_subcategory(src_path, "repo_other_ai_mascot"),
                    "group_id": group_id,
                    "source": rel_orig,
                }
            )

    # 2. Process character-art/v1/download/ (highest-res canonical character art only)
    char_art_dir = RAW_REPO_DIR / "character-art" / "v1" / "download"
    if char_art_dir.exists():
        for src_path in sorted(char_art_dir.iterdir()):
            if src_path.suffix.lower() not in IMAGE_EXTS:
                continue
            is_pos = "deepseek" in src_path.stem.lower()
            dst_dir = OUT_POS_DIR if is_pos else OUT_NEG_AI_DIR
            group_id = f"repo_charart_{src_path.stem}"
            try:
                saved_paths = extract_frames(src_path, dst_dir, group_id, max_frames=1)
            except Exception as exc:
                print(f"[warn] Failed to decode {src_path}: {exc}")
                continue
            for p in saved_paths:
                records.append(
                    {
                        "path": str(p),
                        "label": 1 if is_pos else 0,
                        "category": "positive_dafeiyu" if is_pos else "negative_blue_chars",
                        "subcategory": "repo_charart" if is_pos else mascot_subcategory(src_path, "repo_other_ai_charart"),
                        "group_id": group_id,
                        "source": str(src_path.relative_to(RAW_REPO_DIR)),
                    }
                )

    # 3. Include reviewed originals from mixed-use directories.
    for rel in REVIEWED_POSITIVES:
        src_path = RAW_REPO_DIR / rel
        if not src_path.is_file():
            print(f"[warn] Reviewed positive missing: {src_path}")
            continue
        directory = src_path.parent.name
        group_id = f"repo_extra_{directory}_{src_path.stem}"
        try:
            saved_paths = extract_frames(src_path, OUT_POS_DIR, group_id, max_frames=1)
        except Exception as exc:
            print(f"[warn] Failed to decode {src_path}: {exc}")
            continue
        for p in saved_paths:
            records.append(
                {
                    "path": str(p),
                    "label": 1,
                    "category": "positive_dafeiyu",
                    "subcategory": f"repo_{directory}",
                    "group_id": group_id,
                    "source": rel,
                }
            )

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = sum(1 for r in records if r["label"] == 0)
    print(f"[done] Extracted {n_pos} positive frames and {n_neg} in-repo AI-mascot negative images.")
    print(f"[done] Metadata written to {META_PATH}")


if __name__ == "__main__":
    main()
