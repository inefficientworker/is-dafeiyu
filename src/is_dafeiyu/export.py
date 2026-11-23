"""Export trained Dafeiyu classifier with calibrated sigmoid to ONNX and verify output parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from is_dafeiyu.model import create_classifier


class CalibratedWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, temperature: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x).squeeze(-1)
        return torch.sigmoid(logits / self.temperature)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/dinov2_base/best.pt")
    parser.add_argument("--calibration", type=str, default="artifacts/dinov2_base/calibration.json")
    parser.add_argument("--output", type=str, default="artifacts/dinov2_base/model.onnx")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = create_classifier(backbone=ckpt["backbone"], pretrained=False, img_size=ckpt["image_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    temp = 1.0
    calib_path = Path(args.calibration)
    if calib_path.exists():
        temp = float(json.loads(calib_path.read_text(encoding="utf-8")).get("temperature", 1.0))

    wrapped = CalibratedWrapper(model, temp).eval()
    img_size = int(ckpt["image_size"])
    dummy = torch.randn(1, 3, img_size, img_size)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["probability"],
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
    )

    with torch.no_grad():
        pt_out = wrapped(dummy).numpy()
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"image": dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"[export] Saved ONNX model to {out_path} (max abs diff vs PyTorch: {max_diff:.2e})")
    assert max_diff < 1e-3, f"ONNX parity check failed: max_diff={max_diff}"


if __name__ == "__main__":
    main()
