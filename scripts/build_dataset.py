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
CURATED_META = Path("data/curated/meta.jsonl")
SPLITS_DIR = Path("data/splits")
PHASH_THRESHOLD = 4
N_SPLITS = 10


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
    rows = load_jsonl(REPO_META) + load_jsonl(NEG_META) + load_jsonl(CURATED_META)
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
            r["_hash_bits"] = int(r["phash"], 16)
            valid_rows.append(r)
        except Exception as exc:
            print(f"[warn] Dropping unreadable image {p}: {exc}")

    # 2. Join every pHash and source-group match, including links via dropped duplicates.
    #    Sort positives first so cross-label matches retain the positive.
    valid_rows.sort(key=lambda x: (-x["label"], x["path"]))
    parents = list(range(len(valid_rows)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(i: int, j: int) -> None:
        a, b = sorted((root(i), root(j)))
        parents[b] = a

    group_reps: dict[str, int] = {}
    hash_reps: dict[int, int] = {}
    kept_indices: list[int] = []
    for i, row in enumerate(valid_rows):
        join(i, group_reps.setdefault(row["group_id"], i))
        bits = row["_hash_bits"]
        if bits in hash_reps:
            join(i, hash_reps[bits])
            continue

        cross_label = False
        for seen_bits, j in hash_reps.items():
            if (bits ^ seen_bits).bit_count() <= PHASH_THRESHOLD:
                join(i, j)
                cross_label |= row["label"] != valid_rows[j]["label"]
        hash_reps[bits] = i
        if not cross_label:
            kept_indices.append(i)

    kept_rows = [valid_rows[i] for i in kept_indices]
    for i in kept_indices:
        valid_rows[i]["cluster_id"] = valid_rows[root(i)]["group_id"]
    dropped_dups = len(valid_rows) - len(kept_rows)
    print(f"[build] Kept {len(kept_rows)} images after dropping {dropped_dups} exact/cross-label duplicates.")

    # 3. Stratified Group Split: 10 folds -> 8 train / 1 val / 1 test (80/10/10)
    #    Stratify by `category` (so fish, blue_chars, unrelated, positive are balanced in val/test!)
    #    Group by `cluster_id` (so multi-frames & near-duplicates never leak across splits!)
    gemini_groups = {r["cluster_id"] for r in kept_rows if r["subcategory"] == "negative_gemini"}
    y_strata = [
        r["subcategory"]
        if r["subcategory"] == "negative_gemini" and len(gemini_groups) >= N_SPLITS
        else r["category"]
        for r in kept_rows
    ]
    groups = [r["cluster_id"] for r in kept_rows]

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    folds = list(sgkf.split(kept_rows, y_strata, groups))
    gemini_folds = [
        holdout for _, holdout in folds
        if any(kept_rows[i]["subcategory"] == "negative_gemini" for i in holdout)
    ]
    # Reserve one Gemini fold for training whenever both holdouts can have one.
    holdouts = gemini_folds[:2] if len(gemini_folds) >= 3 else [holdout for _, holdout in folds[:2]]
    test_idx, val_idx = (set(holdout) for holdout in holdouts)
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
        gemini = sum(r["subcategory"] == "negative_gemini" for r in items)
        print(f"[split:{name}] total={len(items)} (pos={pos}, neg={neg}) -> {dict(counts)}")
        print(f"[split:{name}] Gemini negatives={gemini}")


if __name__ == "__main__":
    main()
