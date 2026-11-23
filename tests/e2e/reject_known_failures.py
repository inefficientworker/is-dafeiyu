"""Exercise rejection behavior against real previously failing model artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from is_dafeiyu.evaluate import find_high_precision_threshold, select_operating_point
from is_dafeiyu.export import BrowserCalibratedWrapper, verify_onnx_parity
from is_dafeiyu.model import create_classifier
from is_dafeiyu.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--legacy-model-dir", type=Path, required=True)
    parser.add_argument("--legacy-weighted-model", type=Path, required=True)
    parser.add_argument("--different-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    folder = args.legacy_model_dir
    ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    model, _ = create_classifier(ckpt["backbone"], pretrained=False, img_size=224)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cal = json.loads((folder / "calibration.json").read_text())
    wrapped = BrowserCalibratedWrapper(
        model, cal["temperature"], ckpt["mean"], ckpt["std"]
    ).eval()
    paths = [
        "data/processed/negative_ai_mascots/repo_media_gemini-jiao-cai-suan-cai-niuroumian.png",
        "data/processed/negative_blue_chars/rem_rezero/sbo_6897775.jpg",
        "data/curated/gemini_batch2/HXvxPdFbkAAcyhm.jpg",
    ]
    parity = verify_onnx_parity(
        wrapped,
        folder / "model_browser_fp16.onnx",
        "rgba_uint8",
        224,
        paths,
        0.05,
        tuple(ckpt["mean"]),
        tuple(ckpt["std"]),
        cal["threshold"],
    )
    assert not parity["passed"] and parity["max_abs_probability_difference"] > 0.9
    (out / "known_bad_parity.json").write_text(json.dumps(parity, indent=2))

    # Actual deployed weighted model saturates two negative scores together with
    # the positive score; no probability threshold can meet the requirements.
    import onnxruntime as ort

    from is_dafeiyu.export import image_to_rgba_uint8

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(args.legacy_weighted_model),
        options,
        providers=["CPUExecutionProvider"],
    )
    files = sorted(Path("data/regression/x").glob("*.jpg"))
    scores = np.array(
        [
            float(
                session.run(None, {"image": image_to_rgba_uint8(p, 224)})[0].reshape(
                    -1
                )[0]
            )
            for p in files
        ]
    )
    labels = np.array([int(p.name == "HXzHaTpbIAAzk9B.jpg") for p in files])
    selection = select_operating_point(scores, labels, ["x_regression"] * len(files))
    assert not selection["accepted"]
    try:
        find_high_precision_threshold(scores, labels)
    except ValueError:
        pass
    else:
        raise AssertionError("Infeasible calibration silently fell back")

    # Reject a different checkpoint's calibration, and preserve an existing
    # destination model when export validation cannot establish provenance.
    mismatched = out / "mismatched.json"
    mismatched.write_text(
        json.dumps(
            {**cal, "checkpoint_sha256": file_sha256(args.different_checkpoint)}
        )
    )
    commands = [
        [
            "-m",
            "is_dafeiyu.predict",
            str(files[0]),
            "--device",
            "cpu",
            "--checkpoint",
            str(folder / "best.pt"),
            "--calibration",
            str(mismatched),
        ],
    ]
    target = out / "protected.onnx"
    target.write_bytes((folder / "model_browser_fp16.onnx").read_bytes())
    before = file_sha256(target)
    commands.append(
        [
            "-m",
            "is_dafeiyu.export",
            "--checkpoint",
            str(folder / "best.pt"),
            "--browser",
            "--calib-csv",
            "data/splits/val.csv",
            "--output",
            str(target),
        ]
    )
    logs = []
    for i, command in enumerate(commands):
        result = subprocess.run(
            [sys.executable, *command], check=False, capture_output=True, text=True
        )
        (out / f"rejection-{i}.log").write_text(result.stdout + result.stderr)
        assert result.returncode != 0
        expected_error = (
            "Calibration does not belong" if i == 0 else "checkpoint-bound calibration"
        )
        assert expected_error in result.stderr
        logs.append({"command": command, "exit_code": result.returncode})
    assert file_sha256(target) == before
    report = {
        "passed": True,
        "parity_rejected": parity["max_abs_probability_difference"],
        "infeasible_selection": selection,
        "scores": scores.tolist(),
        "commands": logs,
        "existing_model_sha256_preserved": before,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
