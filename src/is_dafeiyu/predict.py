"""CLI inference script for single image, animated image, or directory of images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from is_dafeiyu.dataset import build_transforms
from is_dafeiyu.model import create_classifier, resolve_device

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def load_frames_rgb(path: Path, max_frames: int = 3) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(path) as img:
        n_frames = getattr(img, "n_frames", 1)
        if n_frames <= 1:
            im = ImageOps.exif_transpose(img)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                frames.append(bg)
            else:
                frames.append(im.convert("RGB"))
            return frames

        indices = list(range(n_frames)) if n_frames <= max_frames else sorted({0, n_frames // 2, n_frames - 1})
        for idx in indices:
            img.seek(idx)
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            frames.append(bg)
    return frames


@torch.no_grad()
def predict_path(
    model: torch.nn.Module,
    transform,
    device: torch.device,
    path: Path,
    temperature: float,
    threshold: float,
) -> dict:
    frames = load_frames_rgb(path)
    tensors = torch.stack([transform(f) for f in frames]).to(device)
    logits = model(tensors).squeeze(-1)
    probs = torch.sigmoid(logits / temperature).float().cpu().tolist()
    if isinstance(probs, float):
        probs = [probs]
    max_prob = float(max(probs))
    return {
        "path": str(path),
        "is_dafeiyu": max_prob >= threshold,
        "probability": round(max_prob, 5),
        "threshold": round(threshold, 4),
        "frames_evaluated": len(frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether image(s) contain Dafeiyu.")
    parser.add_argument("target", type=str, help="Image file or directory")
    parser.add_argument("--checkpoint", type=str, default="artifacts/dinov2_base/best.pt")
    parser.add_argument("--calibration", type=str, default="artifacts/dinov2_base/calibration.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json", action="store_true", help="Output JSON array")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = create_classifier(backbone=ckpt["backbone"], pretrained=False, img_size=ckpt["image_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    temp = 1.0
    thr = 0.5
    calib_path = Path(args.calibration)
    if calib_path.exists():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        temp = float(calib.get("temperature", 1.0))
        thr = float(calib.get("threshold", 0.5))
    if args.threshold is not None:
        thr = args.threshold

    tf = build_transforms(ckpt["image_size"], tuple(ckpt["mean"]), tuple(ckpt["std"]), is_train=False)
    target = Path(args.target)
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    results = [predict_path(model, tf, device, p, temp, thr) for p in files]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            tag = "DAFEIYU" if r["is_dafeiyu"] else "NO     "
            print(f"[{tag}] prob={r['probability']:.4f} (thr={r['threshold']:.3f}) | {r['path']}")


if __name__ == "__main__":
    main()
