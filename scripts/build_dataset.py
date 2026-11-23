"""Build persistent, group-aware train/val/calibration/test splits."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from is_dafeiyu.data_policy import is_real_aquatic
from is_dafeiyu.provenance import file_sha256, relative_path, row_group_keys

REPO_META = Path("data/processed/repo_meta.jsonl")
NEG_META = Path("data/processed/negatives_meta.jsonl")
CURATED_META = Path("data/curated/meta.jsonl")
SPLITS_DIR = Path("data/splits")
PHASH_THRESHOLD = 4
RATIOS = {"train": 0.7, "val": 0.1, "calibration": 0.1, "test": 0.1}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path, nargs="+", default=[REPO_META, NEG_META, CURATED_META]
    )
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument(
        "--real-aquatic-train-limit",
        type=int,
        default=60,
        help="Limit easy aquatic photos in train; retain holdout assignments (-1 keeps all)",
    )
    args = parser.parse_args()
    output = args.splits_dir
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "build_report.json"
    state_path = output / "split_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"version": 1, "assignments": {}, "files": {}}
    )
    assignments = {}
    for key, split in state["assignments"].items():
        key = f"path:{relative_path(key[5:])}" if key.startswith("path:") else key
        if assignments.setdefault(key, split) != split:
            raise ValueError(f"Conflicting historical split assignments: {key}")
    previous_files = {}
    for path, sha256 in state["files"].items():
        path = relative_path(path)
        if previous_files.setdefault(path, sha256) != sha256:
            raise ValueError(f"Conflicting historical file identities: {path}")
    existing_paths = set()
    conflicts = []

    # Bootstrap a legacy split without reassigning its validation/test images.
    for split in RATIOS:
        old_csv = output / f"{split}.csv"
        if not old_csv.exists():
            continue
        with old_csv.open(newline="", encoding="utf-8") as f:
            for old in csv.DictReader(f):
                existing_paths.add(old["path"])
                if Path(old["path"]).exists():
                    old["sha256"] = old.get("sha256") or file_sha256(old["path"])
                    previous_files.setdefault(
                        relative_path(old["path"]), old["sha256"]
                    )
                for key in row_group_keys(old):
                    owner = assignments.setdefault(key, split)
                    if owner != split:
                        conflicts.append(
                            {
                                "kind": "existing_split_overlap",
                                "key": key,
                                "splits": [owner, split],
                            }
                        )

    def fail() -> None:
        audit_path.write_text(
            json.dumps(
                {"accepted": False, "conflicts": conflicts},
                indent=2,
                ensure_ascii=False,
            )
        )
        raise ValueError(
            f"Dataset requires review; existing splits were not rewritten. See {audit_path}"
        )

    rows = []
    by_path = {}
    for source in args.metadata:
        for record in load_jsonl(source):
            row = dict(record)
            p = Path(row["path"])
            row["path"] = relative_path(p)
            row["label"] = int(row["label"])
            if is_real_aquatic(row):
                row["category"] = "negative_unrelated"
            if row["label"] not in (0, 1):
                raise ValueError(f"Invalid label: {row}")
            previous = by_path.get(relative_path(p))
            if previous:
                if any(
                    previous.get(k) != row.get(k)
                    for k in (
                        "label",
                        "group_id",
                        "template_id",
                        "series_id",
                        "author_id",
                    )
                ):
                    conflicts.append({"kind": "conflicting_metadata", "path": str(p)})
                continue
            by_path[relative_path(p)] = row
            try:
                row["sha256"] = file_sha256(p)
                if previous_files.get(relative_path(p), row["sha256"]) != row["sha256"]:
                    conflicts.append({"kind": "changed_image_content", "path": str(p)})
                with Image.open(p) as image:
                    rgba = ImageOps.exif_transpose(image).convert("RGBA")
                    pixel_hash = hashlib.sha256(
                        str(rgba.size).encode() + rgba.tobytes()
                    ).hexdigest()
                    row["pixel_sha256"] = pixel_hash
                    row["phash"] = str(imagehash.phash(rgba.convert("RGB")))
                rows.append(row)
            except (OSError, ValueError) as exc:
                conflicts.append(
                    {"kind": "unreadable_image", "path": str(p), "error": str(exc)}
                )
    if conflicts:
        fail()
    if not rows:
        raise ValueError("No dataset images")
    rows.sort(key=lambda row: row["path"])
    parents = list(range(len(rows)))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(i: int, j: int) -> None:
        a, b = sorted((root(i), root(j)))
        parents[b] = a

    identities = {}
    pixel_reps = {}
    hash_reps = {}
    for i, row in enumerate(rows):
        for key in row_group_keys(row) | {f"pixel_sha256:{row['pixel_sha256']}"}:
            join(i, identities.setdefault(key, i))
        previous = pixel_reps.setdefault(row["pixel_sha256"], i)
        if rows[previous]["label"] != row["label"]:
            conflicts.append(
                {
                    "kind": "contradictory_identical_pixels",
                    "paths": [rows[previous]["path"], row["path"]],
                }
            )
        if (
            row["path"] in existing_paths
            and rows[previous]["path"] not in existing_paths
        ):
            pixel_reps[row["pixel_sha256"]] = i
        bits = int(row["phash"], 16)
        if bits in hash_reps:
            join(i, hash_reps[bits])
        else:
            for other, j in hash_reps.items():
                if (bits ^ other).bit_count() <= PHASH_THRESHOLD:
                    join(i, j)
            hash_reps[bits] = i
        # Near-identical artwork can depict different characters: group, do not relabel/drop it.
    kept = sorted(pixel_reps.values())
    if conflicts:
        fail()

    clusters = collections.defaultdict(list)
    for i in range(len(rows)):
        clusters[root(i)].append(i)
    cluster_keys = {}
    ownership = {}
    strata = {}
    for cluster, indices in clusters.items():
        keys = set().union(
            *(
                row_group_keys(rows[i]) | {f"pixel_sha256:{rows[i]['pixel_sha256']}"}
                for i in indices
            )
        )
        cluster_keys[cluster] = keys
        owners = {assignments[key] for key in keys if key in assignments}
        if len(owners) > 1:
            conflicts.append(
                {
                    "kind": "group_connects_existing_splits",
                    "splits": sorted(owners),
                    "paths": [rows[i]["path"] for i in indices],
                }
            )
        elif owners:
            ownership[cluster] = owners.pop()
        strata[cluster] = {
            rows[i]["subcategory"] if rows[i]["label"] == 0 else "positive_dafeiyu"
            for i in indices
        }
    if conflicts:
        fail()

    total = collections.Counter(s for group in strata.values() for s in group)
    counts = {split: collections.Counter() for split in RATIOS}
    for cluster, split in ownership.items():
        counts[split].update(strata[cluster])
    order = sorted(
        clusters,
        key=lambda c: hashlib.sha256(
            "\n".join(sorted(cluster_keys[c])).encode()
        ).hexdigest(),
    )
    migration = []
    if assignments and not (output / "calibration.csv").exists():
        # One-time reservation from legacy train only. Old models cannot claim this
        # as unseen calibration data; new training records the new manifests.
        for cluster in order:
            group = strata[cluster]
            if ownership.get(cluster) != "train" or any(
                counts["train"][s] <= 1 for s in group
            ):
                continue
            if any(
                counts["calibration"][s] < round(total[s] * RATIOS["calibration"])
                for s in group
            ):
                ownership[cluster] = "calibration"
                counts["train"].subtract(group)
                counts["calibration"].update(group)
                migration.extend(rows[i]["path"] for i in clusters[cluster])
    for cluster in order:
        if cluster in ownership:
            continue
        group = strata[cluster]

        def deficit(split: str, group: set[str] = group) -> float:
            return sum(
                (
                    (
                        1 + (total[s] - 4) * RATIOS[split]
                        if total[s] >= 4
                        else total[s] * RATIOS[split]
                    )
                    - counts[split][s]
                )
                / total[s]
                for s in group
            )

        split = max(RATIOS, key=deficit)
        ownership[cluster] = split
        counts[split].update(group)

    splits = {split: [] for split in RATIOS}
    for cluster, keys in cluster_keys.items():
        for key in keys:
            assignments[key] = ownership[cluster]
    for row in rows:
        previous_files[relative_path(row["path"])] = row["sha256"]
    for i in kept:
        row = rows[i]
        cluster = root(i)
        row["cluster_id"] = rows[cluster]["group_id"]
        splits[ownership[cluster]].append(row)
        assignments[f"group_id:{row['cluster_id']}"] = ownership[cluster]

    # Remove easy training negatives by entire source group, balanced across species.
    # Keep their historical assignments so a future rebuild cannot move them to test.
    aquatic_groups = collections.defaultdict(list)
    for row in splits["train"]:
        aquatic_groups[row["cluster_id"]].append(row)
    eligible = {
        group: records
        for group, records in aquatic_groups.items()
        if all(is_real_aquatic(row) for row in records)
    }
    by_species = collections.defaultdict(list)
    for group, records in eligible.items():
        by_species[min(row["subcategory"] for row in records)].append(group)
    for groups in by_species.values():
        groups.sort(key=lambda group: hashlib.sha256(group.encode()).hexdigest())
    selected = set()
    selected_count = 0
    if args.real_aquatic_train_limit < 0:
        selected.update(eligible)
    else:
        while any(by_species.values()):
            for species in sorted(by_species):
                if not by_species[species]:
                    continue
                group = by_species[species].pop(0)
                if (
                    selected_count + len(eligible[group])
                    <= args.real_aquatic_train_limit
                ):
                    selected.add(group)
                    selected_count += len(eligible[group])
    excluded_aquatic = [
        row["path"]
        for group, records in eligible.items()
        if group not in selected
        for row in records
    ]
    excluded_paths = set(excluded_aquatic)
    splits["train"] = [
        row for row in splits["train"] if row["path"] not in excluded_paths
    ]

    fields = [
        "path",
        "label",
        "category",
        "subcategory",
        "group_id",
        "cluster_id",
        "phash",
        "sha256",
        "pixel_sha256",
        "template_id",
        "series_id",
        "author_id",
        "source",
    ]
    # All checks precede writes; stage all files before replacing them.
    for split, records in splits.items():
        with (output / f"{split}.csv.tmp").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    state = {"version": 2, "assignments": assignments, "files": previous_files}
    state_path.with_suffix(".json.tmp").write_text(
        json.dumps(state, indent=2, ensure_ascii=False)
    )
    for split in RATIOS:
        (output / f"{split}.csv.tmp").replace(output / f"{split}.csv")
    state_path.with_suffix(".json.tmp").replace(state_path)
    report = {
        "accepted": True,
        "calibration_reserved_from_legacy_train": migration,
        "exact_duplicates_removed": len(rows) - len(kept),
        "real_aquatic_train_limit": args.real_aquatic_train_limit,
        "real_aquatic_train_excluded": excluded_aquatic,
        "splits": {},
    }
    for split, records in splits.items():
        report["splits"][split] = {
            "count": len(records),
            "groups": len({r["cluster_id"] for r in records}),
            "subcategories": dict(
                collections.Counter(r["subcategory"] for r in records)
            ),
            "csv_sha256": file_sha256(output / f"{split}.csv"),
        }
        print(f"[split:{split}] {report['splits'][split]}")
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
