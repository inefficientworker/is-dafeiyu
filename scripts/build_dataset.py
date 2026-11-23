"""Deduplicate images via perceptual hash (pHash) and create leak-free Stratified Group splits."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import imagehash
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

REPO_META = Path("data/processed/repo_meta.jsonl")
NEG_META = Path("data/processed/negatives_meta.jsonl")
SPLITS_DIR = Path("data/splits")
PHASH_THRESHOLD = 4


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    rows = load_jsonl(REPO_META) + load_jsonl(NEG_META)
    print(f"[build] Loaded {len(rows)} raw records.")

    # 1. Verify readability & compute pHash
    valid_rows: list[dict] = []
    seen_paths: set[str] = set()
    for r in rows:
        p = Path(r["path"])
        if not p.exists() or str(p) in seen_paths:
            continue
        seen_paths.add(str(p))
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img.convert("RGB"))
            r["phash"] = str(h)
            r["_hash_obj"] = h
            valid_rows.append(r)
        except Exception as exc:
            print(f"[warn] Dropping unreadable image {p}: {exc}")

    # 2. Exact / near-duplicate clustering using Union-Find on pHash + group_id
    #    Keep positives prioritized if a cross-label collision ever occurs (unlikely).
    valid_rows.sort(key=lambda x: (-x["label"], x["path"]))
    kept_rows: list[dict] = []
    kept_hashes: list[tuple[imagehash.ImageHash, int, str]] = []  # (hash, label, cluster_id)
    dropped_dups = 0

    for r in valid_rows:
        h = r["_hash_obj"]
        lbl = r["label"]
        gid = r["group_id"]
        matched_cluster = None
        is_exact_dup = False
        for kh, klbl, kcluster in kept_hashes:
            dist = h - kh
            if dist == 0:
                is_exact_dup = True
                break
            if dist <= PHASH_THRESHOLD:
                if klbl != lbl:
                    # Near-duplicate across opposite labels -> drop the negative
                    is_exact_dup = True
                    break
                matched_cluster = kcluster
                break
        if is_exact_dup:
            dropped_dups += 1
            continue
        cluster_id = matched_cluster or gid
        r["cluster_id"] = cluster_id
        kept_hashes.append((h, lbl, cluster_id))
        kept_rows.append(r)

    print(f"[build] Kept {len(kept_rows)} images after dropping {dropped_dups} exact/cross-label duplicates.")

    # 3. Stratified Group Split: 10 folds -> 8 train / 1 val / 1 test (80/10/10)
    #    Stratify by `category` (so fish, blue_chars, unrelated, positive are balanced in val/test!)
    #    Group by `cluster_id` (so multi-frames & near-duplicates never leak across splits!)
    y_strata = [r["category"] for r in kept_rows]
    groups = [r["cluster_id"] for r in kept_rows]

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(sgkf.split(kept_rows, y_strata, groups))
    test_idx = set(folds[0][1])
    val_idx = set(folds[1][1])
    train_idx = [i for i in range(len(kept_rows)) if i not in test_idx and i not in val_idx]

    splits = {
        "train": [kept_rows[i] for i in train_idx],
        "val": [kept_rows[i] for i in sorted(val_idx)],
        "test": [kept_rows[i] for i in sorted(test_idx)],
    }

    # 4. Assert zero leakage across splits
    split_groups = {name: {r["cluster_id"] for r in items} for name, items in splits.items()}
    assert split_groups["train"].isdisjoint(split_groups["val"]), "Group leakage between train and val!"
    assert split_groups["train"].isdisjoint(split_groups["test"]), "Group leakage between train and test!"
    assert split_groups["val"].isdisjoint(split_groups["test"]), "Group leakage between val and test!"

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "category", "subcategory", "group_id", "cluster_id", "phash"]
    for name, items in splits.items():
        out_csv = SPLITS_DIR / f"{name}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in items:
                writer.writerow({k: r[k] for k in fieldnames})

        counts = collections.Counter(r["category"] for r in items)
        pos = sum(r["label"] for r in items)
        neg = len(items) - pos
        print(f"[split:{name}] total={len(items)} (pos={pos}, neg={neg}) -> {dict(counts)}")


if __name__ == "__main__":
    main()
