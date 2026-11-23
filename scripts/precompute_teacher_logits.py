"""Precompute teacher logits on train and val splits (with horizontal-flip TTA) for distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from is_dafeiyu.dataset import DafeiyuDataset, build_transforms
from is_dafeiyu.model import create_classifier, resolve_amp, resolve_device


@torch.no_grad()
def infer_tta_logits(
    model: torch.nn.Module,
    csv_path: str | Path,
    transform,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[dict[str, float], np.ndarray]:
    ds = DafeiyuDataset(csv_path, transform, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    amp_enabled, amp_dtype = resolve_amp(device, use_amp=True)

    logits_map: dict[str, float] = {}
    all_logits: list[float] = []
    all_labels: list[int] = []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        imgs_flip = TF.hflip(imgs)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits_orig = model(imgs).squeeze(-1)
            logits_flip = model(imgs_flip).squeeze(-1)
        avg_logits = (0.5 * (logits_orig.float() + logits_flip.float())).cpu().numpy()
        for p, logit_val, lbl in zip(batch["path"], avg_logits.tolist(), batch["label"].tolist()):
            logits_map[p] = float(logit_val)
            all_logits.append(float(logit_val))
            all_labels.append(int(lbl))

    preds = (np.array(all_logits) >= 0.0).astype(int)
    acc = float((preds == np.array(all_labels)).mean())
    print(f"[precompute] {csv_path}: {len(logits_map)} samples, teacher raw-logit acc@0={acc:.4f}")
    return logits_map, np.array(all_logits, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/dinov2_base/best.pt")
    parser.add_argument("--train-csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--output", type=str, default="artifacts/dinov2_base/teacher_logits.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = resolve_device("auto")
    print(f"[precompute] Using device: {device}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = create_classifier(backbone=ckpt["backbone"], pretrained=False, img_size=ckpt["image_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    eval_tf = build_transforms(ckpt["image_size"], tuple(ckpt["mean"]), tuple(ckpt["std"]), is_train=False)
    train_map, train_arr = infer_tta_logits(model, args.train_csv, eval_tf, device, batch_size=args.batch_size)
    val_map, val_arr = infer_tta_logits(model, args.val_csv, eval_tf, device, batch_size=args.batch_size)

    merged = {**train_map, **val_map}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "teacher_backbone": ckpt["backbone"],
        "image_size": int(ckpt["image_size"]),
        "train_count": len(train_map),
        "val_count": len(val_map),
        "logit_range": [float(min(train_arr.min(), val_arr.min())), float(max(train_arr.max(), val_arr.max()))],
        "logits": merged,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[precompute] Saved {len(merged)} teacher logits to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
