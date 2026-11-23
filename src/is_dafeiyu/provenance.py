"""Content identities and dataset/model provenance shared by the training pipeline."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def relative_path(path: str | Path) -> str:
    """Persist canonical paths relative to the working dataset root."""
    return Path(os.path.relpath(Path(path).resolve(), Path.cwd())).as_posix()


def row_group_keys(row: dict) -> set[str]:
    keys = {f"path:{relative_path(row['path'])}"}
    for name in (
        "group_id",
        "cluster_id",
        "template_id",
        "series_id",
        "author_id",
        "sha256",
        "pixel_sha256",
    ):
        if row.get(name):
            namespace = "group_id" if name == "cluster_id" else name
            keys.add(f"{namespace}:{row[name]}")
    return keys


def split_provenance(paths: dict[str, str | Path]) -> dict:
    """Reject overlap before any training/calibration, including copied image bytes."""
    owners: dict[str, str] = {}
    result = {}
    for split, path in paths.items():
        path = Path(path)
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty {split} split: {path}")
        for row in rows:
            actual_sha = file_sha256(row["path"])
            if row.get("sha256") and row["sha256"] != actual_sha:
                raise ValueError(
                    f"Image content changed since split creation: {row['path']}"
                )
            row["sha256"] = actual_sha
            for key in row_group_keys(row):
                previous = owners.setdefault(key, split)
                if previous != split:
                    raise ValueError(
                        f"Dataset leakage: {key} occurs in {previous} and {split}"
                    )
        result[split] = {
            "path": relative_path(path),
            "csv_sha256": file_sha256(path),
            "count": len(rows),
        }
    return result


def configured_splits(cfg: dict) -> dict[str, str]:
    return {
        name: cfg["data"][f"{name}_csv"]
        for name in ("train", "val", "calibration", "test")
    }


def check_checkpoint_splits(checkpoint: dict, current: dict) -> None:
    recorded = checkpoint.get("data_provenance")
    if recorded is None:
        raise ValueError(
            "Checkpoint has no split provenance; retrain with the versioned splits before calibration."
        )
    for split, info in current.items():
        if recorded.get(split, {}).get("csv_sha256") != info["csv_sha256"]:
            raise ValueError(
                f"Checkpoint was trained against a different {split} manifest"
            )


def check_model_calibration(
    calibration: dict, model_path: str | Path, key: str = "checkpoint_sha256"
) -> None:
    expected = calibration.get(key)
    if expected is not None and expected != file_sha256(model_path):
        raise ValueError(f"Calibration does not belong to this model: {model_path}")
    if calibration.get("accepted") is False:
        raise ValueError(
            "Calibration did not meet the configured operating constraints"
        )
