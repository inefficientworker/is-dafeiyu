"""Export trained Dafeiyu classifier (optional browser RGBA uint8 wrapper + quantization) to single-file ONNX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from torch import nn

from is_dafeiyu.dataset import build_transforms, load_rgb_image
from is_dafeiyu.evaluate import operating_constraints, select_operating_point
from is_dafeiyu.model import create_classifier
from is_dafeiyu.provenance import check_model_calibration, file_sha256


class CalibratedWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, temperature: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x).squeeze(-1)
        return torch.sigmoid(logits / self.temperature)


class BrowserCalibratedWrapper(nn.Module):
    """End-to-end browser wrapper accepting RGBA uint8 [B, H, W, 4] and outputting probability [B]."""

    def __init__(
        self,
        backbone: nn.Module,
        temperature: float,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.temperature = float(temperature)
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.to(torch.float32)
        rgb = x_f[:, :, :, :3]
        alpha = x_f[:, :, :, 3:4] * (1.0 / 255.0)
        composited = rgb * alpha + 255.0 * (1.0 - alpha)
        nchw = composited.permute(0, 3, 1, 2) * (1.0 / 255.0)
        normalized = (nchw - self.mean) / self.std
        logits = self.backbone(normalized).squeeze(-1)
        return torch.sigmoid(logits / self.temperature)


def image_to_rgba_uint8(path: str | Path, image_size: int) -> np.ndarray:
    """Load an image, apply EXIF orientation, resize to image_size, and return RGBA uint8 [1, H, W, 4]."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGBA")
        resized = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.uint8)
    return arr[None, ...]


def save_single_file_onnx(src_path: Path, dst_path: Path) -> None:
    """Ensure the ONNX graph and all weights are stored in a single .onnx file without external .data."""
    model = onnx.load(str(src_path), load_external_data=True)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(dst_path), save_as_external_data=False)
    for leftover in (
        dst_path.with_name(dst_path.name + ".data"),
        src_path.with_name(src_path.name + ".data"),
    ):
        if leftover.exists() and leftover != dst_path:
            leftover.unlink()


def export_fp32_onnx(
    wrapped: nn.Module,
    dummy: torch.Tensor,
    out_path: Path,
) -> None:
    tmp_path = out_path.with_name(out_path.stem + ".__tmp__.onnx")
    try:
        export_kwargs = {
            "input_names": ["image"],
            "output_names": ["probability"],
            "dynamic_axes": {"image": {0: "batch"}, "probability": {0: "batch"}},
            "opset_version": 17,
        }
        try:
            torch.onnx.export(
                wrapped, dummy, str(tmp_path), dynamo=False, **export_kwargs
            )
        except TypeError:
            torch.onnx.export(wrapped, dummy, str(tmp_path), **export_kwargs)
        save_single_file_onnx(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        tmp_data = tmp_path.with_name(tmp_path.name + ".data")
        if tmp_data.exists():
            tmp_data.unlink()


def apply_quantization(
    fp32_path: Path,
    out_path: Path,
    mode: str,
    input_format: str,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    calib_csv: str | Path,
    calib_samples: int,
) -> None:
    if mode == "fp16":
        from onnxruntime.transformers import float16

        model = onnx.load(str(fp32_path), load_external_data=True)
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=False,
        )
        onnx.save_model(model_fp16, str(out_path), save_as_external_data=False)
        return

    if mode == "int8_dynamic":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        tmp_q = out_path.with_name(out_path.stem + ".__qtmp__.onnx")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(tmp_q),
            weight_type=QuantType.QUInt8,
        )
        save_single_file_onnx(tmp_q, out_path)
        tmp_q.unlink(missing_ok=True)
        return

    if mode == "int8_static":
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quant_pre_process,
            quantize_static,
        )

        with Path(calib_csv).open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rng = np.random.default_rng(42)
        if len(rows) > calib_samples:
            idxs = rng.choice(len(rows), size=calib_samples, replace=False)
            rows = [rows[int(i)] for i in sorted(idxs)]

        eval_tf = build_transforms(image_size, mean, std, is_train=False)

        class _Reader(CalibrationDataReader):
            def __init__(self) -> None:
                self._idx = 0

            def get_next(self) -> dict | None:
                if self._idx >= len(rows):
                    return None
                p = rows[self._idx]["path"]
                self._idx += 1
                if input_format == "rgba_uint8":
                    arr = image_to_rgba_uint8(p, image_size)
                else:
                    rgb = load_rgb_image(p, random_bg=False)
                    arr = eval_tf(rgb).unsqueeze(0).numpy().astype(np.float32)
                return {"image": arr}

        prep_path = out_path.with_name(out_path.stem + ".__prep__.onnx")
        tmp_q = out_path.with_name(out_path.stem + ".__qtmp__.onnx")
        try:
            quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)
            model_for_quant = prep_path if prep_path.exists() else fp32_path
            quantize_static(
                model_input=str(model_for_quant),
                model_output=str(tmp_q),
                calibration_data_reader=_Reader(),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
                per_channel=True,
                op_types_to_quantize=["Conv", "Gemm", "MatMul"],
            )
            save_single_file_onnx(tmp_q, out_path)
        finally:
            prep_path.unlink(missing_ok=True)
            tmp_q.unlink(missing_ok=True)
        return

    raise ValueError(f"Unknown quantization mode: {mode}")


def verify_onnx_parity(
    wrapped: nn.Module,
    onnx_path: Path,
    input_format: str,
    image_size: int,
    sample_paths: list[str],
    tol: float,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    threshold: float = 0.5,
) -> dict:
    """Check every calibration image and real multi-image batches on the final artifact."""
    if not sample_paths:
        raise ValueError("Export validation requires a nonempty calibration split")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    sess = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    tf = build_transforms(image_size, mean, std, is_train=False)
    inputs = [
        image_to_rgba_uint8(p, image_size).copy()
        if input_format == "rgba_uint8"
        else tf(load_rgb_image(p)).unsqueeze(0).numpy()
        for p in sample_paths
    ]
    results = []
    with torch.inference_mode():
        for p, arr in zip(sample_paths, inputs):
            pt = float(wrapped(torch.from_numpy(arr)).item())
            onnx_prob = float(sess.run(None, {"image": arr})[0].reshape(-1)[0])
            results.append(
                {"path": p, "pytorch_probability": pt, "onnx_probability": onnx_prob}
            )
        for start in range(0, len(inputs), 8):
            batch = np.concatenate(inputs[start : start + 8])
            pt = wrapped(torch.from_numpy(batch)).numpy().reshape(-1)
            onnx_probs = sess.run(None, {"image": batch})[0].reshape(-1)
            for row, p, q in zip(results[start : start + 8], pt, onnx_probs):
                row["pytorch_batch_probability"] = float(p)
                row["onnx_batch_probability"] = float(q)
    values = np.array(
        [
            [
                r[k]
                for k in (
                    "pytorch_probability",
                    "onnx_probability",
                    "pytorch_batch_probability",
                    "onnx_batch_probability",
                )
            ]
            for r in results
        ]
    )
    finite = bool(np.isfinite(values).all())
    max_diff = float(np.max(np.abs(values[:, 0] - values[:, 1]))) if finite else None
    max_batch_diff = (
        float(
            max(
                np.max(np.abs(values[:, 0] - values[:, 2])),
                np.max(np.abs(values[:, 1] - values[:, 3])),
            )
        )
        if finite
        else None
    )
    valid_range = finite and bool(((values >= -1e-6) & (values <= 1 + 1e-6)).all())
    return {
        "passed": bool(valid_range and max_diff <= tol and max_batch_diff <= tol),
        "tolerance": tol,
        "max_abs_probability_difference": max_diff,
        "max_batch_probability_difference": max_batch_diff,
        "decision_differences_at_pytorch_threshold": int(
            ((values[:, 0] >= threshold) != (values[:, 1] >= threshold)).sum()
        ),
        "samples": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="artifacts/dinov2_base_v2/best.pt"
    )
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--browser", action="store_true", help="Shorthand for --input-format rgba_uint8"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["nchw_float32", "rgba_uint8"],
        default=None,
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "fp16", "int8_dynamic", "int8_static"],
        default="none",
    )
    parser.add_argument("--calib-csv", type=str, default="data/splits/calibration.csv")
    parser.add_argument("--calib-samples", type=int, default=200)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    calib_path = (
        Path(args.calibration)
        if args.calibration
        else ckpt_path.parent / "calibration.json"
    )
    input_format = args.input_format or (
        "rgba_uint8" if args.browser else "nchw_float32"
    )
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = ""
        if input_format == "rgba_uint8":
            suffix += "_browser"
        if args.quantize != "none":
            suffix += f"_{args.quantize}"
        out_path = ckpt_path.parent / f"model{suffix}.onnx"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, _ = create_classifier(
        backbone=ckpt["backbone"], pretrained=False, img_size=ckpt["image_size"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    check_model_calibration(calib, ckpt_path)
    expected_csv = (
        calib.get("data_provenance", {}).get("calibration", {}).get("csv_sha256")
    )
    if "checkpoint_sha256" not in calib or expected_csv != file_sha256(args.calib_csv):
        raise ValueError(
            "Export requires checkpoint-bound calibration from the independent calibration split"
        )
    with Path(args.calib_csv).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("sha256") != file_sha256(row["path"]):
            raise ValueError(
                f"Calibration image changed since split creation: {row['path']}"
            )
    temp = float(calib["temperature"])

    image_size = int(ckpt["image_size"])
    mean = tuple(float(v) for v in ckpt["mean"])
    std = tuple(float(v) for v in ckpt["std"])

    if input_format == "rgba_uint8":
        wrapped: nn.Module = BrowserCalibratedWrapper(model, temp, mean, std).eval()
        dummy = torch.randint(0, 256, (1, image_size, image_size, 4), dtype=torch.uint8)
    else:
        wrapped = CalibratedWrapper(model, temp).eval()
        dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = out_path.with_name(out_path.stem + ".__candidate__.onnx")
    if args.quantize == "none":
        export_fp32_onnx(wrapped, dummy, candidate)
    else:
        fp32_tmp = candidate.with_name(candidate.stem + ".__fp32__.onnx")
        try:
            export_fp32_onnx(wrapped, dummy, fp32_tmp)
            apply_quantization(
                fp32_tmp,
                candidate,
                args.quantize,
                input_format,
                image_size,
                mean,
                std,
                args.calib_csv,
                args.calib_samples,
            )
        finally:
            fp32_tmp.unlink(missing_ok=True)

    tol = (
        1e-3 if args.quantize == "none" else (5e-2 if args.quantize == "fp16" else 0.15)
    )
    report = verify_onnx_parity(
        wrapped,
        candidate,
        input_format,
        image_size,
        [r["path"] for r in rows],
        tol,
        mean,
        std,
        float(calib["threshold"]),
    )
    report["accepted"] = False
    report["candidate_sha256"] = file_sha256(candidate)
    report_path = out_path.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise ValueError(
            f"Export parity failed; existing model preserved. See {report_path}"
        )
    scores = np.clip([r["onnx_probability"] for r in report["samples"]], 0, 1)
    selection = select_operating_point(
        scores,
        np.array([int(r["label"]) for r in rows]),
        [r["subcategory"] for r in rows],
        **operating_constraints(calib["constraints"]),
    )
    report["deployment_selection"] = selection
    if selection["accepted"]:
        threshold = selection["operating_point"]["threshold"]
        report["batch_decision_differences"] = sum(
            (r["onnx_probability"] >= threshold)
            != (r["onnx_batch_probability"] >= threshold)
            for r in report["samples"]
        )
        report["accepted"] = report["batch_decision_differences"] == 0
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["accepted"]:
        raise ValueError(
            f"Deployed model did not meet calibration/batch-stability constraints. See {report_path}"
        )
    deployment = {
        "accepted": True,
        "threshold": threshold,
        "baked_temperature": temp,
        "threshold_backend": "onnx_final_artifact",
        "input_format": input_format,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "onnx_sha256": report["candidate_sha256"],
        "checkpoint_sha256": file_sha256(ckpt_path),
        "data_provenance": calib["data_provenance"],
        **selection,
    }
    sidecar = out_path.with_suffix(".calibration.json")
    sidecar.with_suffix(".json.tmp").write_text(
        json.dumps(deployment, indent=2), encoding="utf-8"
    )
    candidate.replace(out_path)
    sidecar.with_suffix(".json.tmp").replace(sidecar)
    print(
        f"[export] Accepted {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB); threshold={threshold:.8f}"
    )
    print(f"[export] Calibration: {sidecar}; full parity report: {report_path}")


if __name__ == "__main__":
    main()
