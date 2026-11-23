"""Real-data CLI pipeline acceptance. Run with uv run python tests/e2e/training_pipeline.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from is_dafeiyu.provenance import file_sha256

SPLITS = ("train", "val", "calibration", "test")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.output
    if out.exists() and not args.resume:
        raise ValueError(
            "Choose a fresh output directory, or --resume after inspecting the logs"
        )
    out.mkdir(parents=True, exist_ok=True)
    commands = []
    report = {
        "checks": {},
        "commands": commands,
        "production_split_hashes_before": {
            name: file_sha256(f"data/splits/{name}.csv")
            for name in ("train", "val", "test")
        },
    }
    report_path = out / "report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    def run(name, args, expect=0):
        command = [sys.executable, *args]
        log = out / (name + ".log")
        if args_resume and log.exists() and name in completed:
            return
        with log.open("w") as f:
            status = subprocess.run(
                command,
                check=False,
                stdout=f,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "OMP_NUM_THREADS": "4",
                    "PYTHONUNBUFFERED": "1",
                },
            ).returncode
        commands.append(
            {"name": name, "command": command, "exit_code": status, "log": str(log)}
        )
        save()
        if (status == 0) != (expect == 0):
            raise AssertionError(f"Unexpected exit {status}; see {log}")
        print(name, "exit", status, flush=True)

    args_resume = args.resume
    completed = set()
    if args.resume and report_path.exists():
        old = json.loads(report_path.read_text())
        completed = {r["name"] for r in old["commands"] if r["exit_code"] == 0}
        report = old
        commands = report["commands"]
    split_dir = out / "splits"
    split_dir.mkdir(exist_ok=True)
    if not (split_dir / "split_state.json").exists():
        for name in ("train", "val", "test"):
            shutil.copyfile(f"data/splits/{name}.csv", split_dir / (name + ".csv"))
    run("build_initial", ["scripts/build_dataset.py", "--splits-dir", str(split_dir)])
    before = {
        name: {r["path"] for r in read_csv(split_dir / (name + ".csv"))}
        for name in SPLITS
    }
    for name in ("val", "test"):
        original = {r["path"] for r in read_csv(f"data/splits/{name}.csv")}
        assert original <= before[name]
    stable_hashes = {name: file_sha256(split_dir / (name + ".csv")) for name in SPLITS}
    run("build_repeat", ["scripts/build_dataset.py", "--splits-dir", str(split_dir)])
    assert stable_hashes == {
        name: file_sha256(split_dir / (name + ".csv")) for name in SPLITS
    }
    report["checks"]["rebuild_stable"] = True

    # Use X.zip only in a separate splitter test, never the training experiment.
    incremental = out / "incremental"
    if not incremental.exists():
        shutil.copytree(split_dir, incremental)
    xdir = Path("data/regression/x")
    extra = out / "extra.jsonl"
    extra.write_text(
        "".join(
            json.dumps(
                {
                    "path": str(p),
                    "label": int(p.name == "HXzHaTpbIAAzk9B.jpg"),
                    "category": "positive_dafeiyu"
                    if p.name == "HXzHaTpbIAAzk9B.jpg"
                    else "negative_blue_chars",
                    "subcategory": "x_diagnostic",
                    "group_id": "x_shared_template",
                    "template_id": "x_shared_template",
                }
            )
            + "\n"
            for p in sorted(xdir.glob("*.jpg"))
        )
    )
    metadata = [
        "data/processed/repo_meta.jsonl",
        "data/processed/negatives_meta.jsonl",
        "data/curated/meta.jsonl",
    ]
    run(
        "build_incremental",
        [
            "scripts/build_dataset.py",
            "--splits-dir",
            str(incremental),
            "--metadata",
            *metadata,
            str(extra),
        ],
    )
    after = {
        r["path"]: name
        for name in SPLITS
        for r in read_csv(incremental / (name + ".csv"))
    }
    assert all(after[p] == name for name, paths in before.items() for p in paths)
    assert len({after[str(p)] for p in xdir.glob("*.jpg")}) == 1
    report["checks"]["incremental_preserves_assignments_and_templates"] = True

    # Copy real pixels with the opposite label; splitting must fail before rewriting.
    conflict_row = dict(read_csv(split_dir / "train.csv")[0])
    original = Path(conflict_row["path"])
    duplicate = out / ("conflicting_copy" + original.suffix)
    shutil.copyfile(original, duplicate)
    conflict_row.update(
        path=str(duplicate),
        label=1 - int(conflict_row["label"]),
        group_id="conflicting_copy",
    )
    conflict = out / "conflict.jsonl"
    conflict.write_text(json.dumps(conflict_row) + "\n")
    run(
        "contradictory_pixels_rejected",
        [
            "scripts/build_dataset.py",
            "--splits-dir",
            str(split_dir),
            "--metadata",
            *metadata,
            str(conflict),
        ],
        expect=1,
    )
    assert stable_hashes == {
        name: file_sha256(split_dir / (name + ".csv")) for name in SPLITS
    }
    assert (
        "contradictory_identical_pixels"
        in (split_dir / "build_report.json").read_text()
    )
    report["checks"]["contradictory_pixels_do_not_rewrite_splits"] = True

    # Existing real train and test images linked by a newly discovered template.
    bridge_rows = [
        dict(read_csv(split_dir / (name + ".csv"))[0]) for name in ("train", "test")
    ]
    bridge_paths = {r["path"] for r in bridge_rows}
    bridge_meta = out / "bridge.jsonl"
    all_rows = [
        json.loads(line)
        for p in metadata
        for line in Path(p).read_text().splitlines()
        if line.strip()
    ]
    for row in all_rows:
        if row["path"] in bridge_paths:
            row["template_id"] = "newly_discovered_common_template"
    bridge_meta.write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    run(
        "cross_split_template_rejected",
        [
            "scripts/build_dataset.py",
            "--splits-dir",
            str(split_dir),
            "--metadata",
            str(bridge_meta),
        ],
        expect=1,
    )
    assert stable_hashes == {
        name: file_sha256(split_dir / (name + ".csv")) for name in SPLITS
    }
    assert (
        "group_connects_existing_splits"
        in (split_dir / "build_report.json").read_text()
    )
    report["checks"]["cross_split_template_conflict"] = True
    save()

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    for name in SPLITS:
        cfg["data"][name + "_csv"] = str(split_dir / (name + ".csv"))
    cfg["data"]["num_workers"] = 2
    cfg["training"].update(
        output_dir=str(out / "model"), stage1_epochs=2, stage2_epochs=3
    )
    config = out / "train.yaml"
    config.write_text(yaml.safe_dump(cfg))
    # Full real dataset, short optimization run; strict production constraints retained.
    run("train", ["-m", "is_dafeiyu.train", "--config", str(config)])
    audit = json.loads((out / "model/training_audit.json").read_text())
    assert all(
        stage["frozen_before"] == stage["frozen_after"] for stage in audit.values()
    )
    assert all(
        h["evaluation_precision"] == "fp32"
        for h in json.loads((out / "model/history.json").read_text())
    )
    report["checks"]["real_two_stage_freeze_and_fp32_selection"] = audit
    run("calibrate", ["-m", "is_dafeiyu.evaluate", "--config", str(config)])
    model = out / "model/model_browser_fp16.onnx"
    run(
        "export",
        [
            "-m",
            "is_dafeiyu.export",
            "--checkpoint",
            str(out / "model/best.pt"),
            "--browser",
            "--quantize",
            "fp16",
            "--calib-csv",
            str(split_dir / "calibration.csv"),
            "--output",
            str(model),
        ],
    )
    run(
        "onnx_test",
        [
            "scripts/eval_onnx.py",
            "--onnx",
            str(model),
            "--test-csv",
            str(split_dir / "test.csv"),
            "--out-json",
            str(out / "onnx-test.json"),
        ],
    )
    run(
        "predict_x",
        [
            "-m",
            "is_dafeiyu.predict",
            str(xdir),
            "--checkpoint",
            str(out / "model/best.pt"),
            "--calibration",
            str(out / "model/calibration.json"),
            "--json",
        ],
    )
    report["checks"]["actual_onnx_threshold"] = json.loads(
        model.with_suffix(".calibration.json").read_text()
    )
    report["production_split_hashes_after"] = {
        name: file_sha256(f"data/splits/{name}.csv")
        for name in ("train", "val", "test")
    }
    assert (
        report["production_split_hashes_before"]
        == report["production_split_hashes_after"]
    )
    report["passed"] = True
    report["scope"] = (
        "Pipeline behavior only; short-run accuracy is not a generalization claim."
    )
    save()
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
