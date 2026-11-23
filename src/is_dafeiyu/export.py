"""Export trained Dafeiyu classifier (optional browser RGBA uint8 wrapper + quantization) to single-file ONNX."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageOps
from torch import nn

from is_dafeiyu.dataset import build_transforms, load_rgb_image
from is_dafeiyu.model import create_classifier


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
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

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
    for leftover in (dst_path.with_name(dst_path.name + ".data"), src_path.with_name(src_path.name + ".data")):
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
            torch.onnx.export(wrapped, dummy, str(tmp_path), dynamo=False, **export_kwargs)
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

        rows = list(csv.DictReader(Path(calib_csv).open("r", encoding="utf-8")))
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
) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = 0.0

    # Dummy check
    if input_format == "rgba_uint8":
        dummy_np = np.random.default_rng(0).integers(0, 256, size=(2, image_size, image_size, 4), dtype=np.uint8)
        dummy_t = torch.from_numpy(dummy_np)
    else:
        dummy_t = torch.randn(2, 3, image_size, image_size, dtype=torch.float32)
        dummy_np = dummy_t.numpy()

    with torch.no_grad():
        pt_out = wrapped(dummy_t).numpy()
    ort_out = sess.run(None, {"image": dummy_np})[0]
    max_diff = max(max_diff, float(np.max(np.abs(pt_out - ort_out))))

    # Real image checks (if available)
    if input_format == "rgba_uint8" and sample_paths:
        for p in sample_paths:
            arr = image_to_rgba_uint8(p, image_size)
            with torch.no_grad():
                pt_p = float(wrapped(torch.from_numpy(arr)).item())
            ort_p = float(sess.run(None, {"image": arr})[0][0])
            diff = abs(pt_p - ort_p)
            max_diff = max(max_diff, diff)
            print(f"  [parity] {Path(p).name}: PyTorch={pt_p:.5f}, ONNX={ort_p:.5f}, diff={diff:.2e}")

    print(f"[export] Max abs diff vs PyTorch: {max_diff:.2e} (tol={tol:.2e})")
    if max_diff > tol:
        raise AssertionError(f"ONNX parity check failed: max_diff={max_diff} > tol={tol}")
    return max_diff


def find_sample_images(csv_path: str | Path) -> list[str]:
    p = Path(csv_path)
    if not p.exists():
        return []
    rows = list(csv.DictReader(p.open("r", encoding="utf-8")))
    picked: list[str] = []
    # Try to pick 1 transparent PNG and 1 opaque/RGB image
    for r in rows:
        img_p = Path(r["path"])
        if not img_p.exists():
            continue
        if img_p.suffix.lower() == ".png" and len(picked) == 0:
            with Image.open(img_p) as im:
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    picked.append(str(img_p))
        elif img_p.suffix.lower() in (".jpg", ".jpeg") and len(picked) == 1:
            picked.append(str(img_p))
        if len(picked) >= 2:
            break
    if not picked and rows:
        for r in rows[:2]:
            if Path(r["path"]).exists():
                picked.append(r["path"])
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/dinov2_base/best.pt")
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--browser", action="store_true", help="Shorthand for --input-format rgba_uint8")
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
    parser.add_argument("--calib-csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--calib-samples", type=int, default=200)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    calib_path = Path(args.calibration) if args.calibration else ckpt_path.parent / "calibration.json"
    input_format = args.input_format or ("rgba_uint8" if args.browser else "nchw_float32")
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
    model, _ = create_classifier(backbone=ckpt["backbone"], pretrained=False, img_size=ckpt["image_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    temp = 1.0
    if calib_path.exists():
        temp = float(json.loads(calib_path.read_text(encoding="utf-8")).get("temperature", 1.0))

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
    if args.quantize == "none":
        export_fp32_onnx(wrapped, dummy, out_path)
    else:
        fp32_tmp = out_path.with_name(out_path.stem + ".__fp32__.onnx")
        try:
            export_fp32_onnx(wrapped, dummy, fp32_tmp)
            apply_quantization(
                fp32_path=fp32_tmp,
                out_path=out_path,
                mode=args.quantize,
                input_format=input_format,
                image_size=image_size,
                mean=mean,
                std=std,
                calib_csv=args.calib_csv,
                calib_samples=args.calib_samples,
            )
        finally:
            fp32_tmp.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    has_ext_data = out_path.with_name(out_path.name + ".data").exists()
    print(f"[export] Saved single-file ONNX ({input_format}, quant={args.quantize}): {out_path} ({size_mb:.2f} MB, external_data={has_ext_data})")
    assert not has_ext_data, "External .data file must not exist!"

    tol = 1e-3 if args.quantize == "none" else (5e-2 if args.quantize == "fp16" else 0.15)
    sample_paths = find_sample_images(args.calib_csv)
    t0 = time.perf_counter()
    verify_onnx_parity(wrapped, out_path, input_format, image_size, sample_paths, tol=tol)
    _ = time.perf_counter() - t0


if __name__ == "__main__":
    main()
