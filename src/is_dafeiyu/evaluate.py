"""Calibrate temperature & high-precision threshold on Val, then evaluate on Test."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from is_dafeiyu.dataset import DafeiyuDataset, build_transforms
from is_dafeiyu.model import create_classifier, resolve_amp, resolve_device


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    logits_list, labels_list, cats, subcats, paths = [], [], [], [], []
    amp_enabled, amp_dtype = resolve_amp(device, use_amp)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images).squeeze(-1)
        logits_list.append(logits.float().cpu().numpy())
        labels_list.append(batch["label"].numpy())
        cats.extend(batch["category"])
        subcats.extend(batch["subcategory"])
        paths.extend(batch["path"])
    return {
        "logits": np.concatenate(logits_list),
        "labels": np.concatenate(labels_list).astype(int),
        "categories": cats,
        "subcategories": subcats,
        "paths": paths,
    }


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Find scalar temperature T > 0 minimizing NLL on validation set."""
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.float32)
    log_temp = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)
    bce = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(0.2, 10.0)
        loss = bce(logits_t / temp, labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_temp).clamp(0.2, 10.0).item())


def find_high_precision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.98,
    f_beta: float = 0.5,
) -> float:
    """Choose threshold prioritizing low false positives (high precision) with max safety margin."""
    candidates = np.linspace(0.30, 0.98, 137)
    best_recall = -1.0
    best_thrs: list[float] = []

    # Find all thresholds meeting target_precision that achieve maximum recall
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        if prec >= target_precision:
            if rec > best_recall + 1e-9:
                best_recall = rec
                best_thrs = [float(thr)]
            elif abs(rec - best_recall) <= 1e-9:
                best_thrs.append(float(thr))

    if best_thrs:
        # Pick the midpoint of the optimal plateau (at least 0.50) for stronger OOD false-positive margin
        mid = 0.5 * (best_thrs[0] + best_thrs[-1])
        return float(max(0.50, mid))

    # Fallback: maximize F_beta (beta=0.5 weights precision 2x more than recall)
    best_thr = 0.5
    best_f = -1.0
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        score = fbeta_score(labels, preds, beta=f_beta, zero_division=0)
        if score > best_f:
            best_f = score
            best_thr = float(thr)
    return best_thr


def compute_detailed_metrics(data: dict, temp: float, threshold: float) -> dict:
    probs = 1.0 / (1.0 + np.exp(-(data["logits"] / temp)))
    labels = data["labels"]
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    by_cat: dict[str, dict] = {}
    cat_indices: dict[str, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(data["categories"]):
        cat_indices[c].append(i)

    for cat, idxs in sorted(cat_indices.items()):
        c_labels = labels[idxs]
        c_preds = preds[idxs]
        if (c_labels == 0).all():
            c_fp = int((c_preds == 1).sum())
            by_cat[cat] = {"count": len(idxs), "false_positives": c_fp, "fpr": float(c_fp / len(idxs))}
        else:
            c_fn = int((c_preds == 0).sum())
            by_cat[cat] = {"count": len(idxs), "false_negatives": c_fn, "recall": float(1.0 - c_fn / len(idxs))}

    false_positives = []
    false_negatives = []
    for i in range(len(labels)):
        if labels[i] == 0 and preds[i] == 1:
            false_positives.append(
                {
                    "path": data["paths"][i],
                    "category": data["categories"][i],
                    "subcategory": data["subcategories"][i],
                    "prob": float(probs[i]),
                }
            )
        elif labels[i] == 1 and preds[i] == 0:
            false_negatives.append(
                {
                    "path": data["paths"][i],
                    "category": data["categories"][i],
                    "subcategory": data["subcategories"][i],
                    "prob": float(probs[i]),
                }
            )
    false_positives.sort(key=lambda x: -x["prob"])
    false_negatives.sort(key=lambda x: x["prob"])

    return {
        "temperature": temp,
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "f0.5": float(fbeta_score(labels, preds, beta=0.5, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "by_category": by_cat,
        "top_false_positives": false_positives[:20],
        "top_false_negatives": false_negatives[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["training"]["output_dir"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best.pt"

    device = resolve_device(cfg["training"]["device"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, _ = create_classifier(backbone=ckpt["backbone"], pretrained=False, img_size=ckpt["image_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    eval_tf = build_transforms(ckpt["image_size"], tuple(ckpt["mean"]), tuple(ckpt["std"]), is_train=False)
    val_ds = DafeiyuDataset(cfg["data"]["val_csv"], eval_tf, is_train=False)
    test_ds = DafeiyuDataset(cfg["data"]["test_csv"], eval_tf, is_train=False)

    num_workers = int(cfg["data"].get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=num_workers)

    use_bf16 = bool(cfg["training"]["use_bf16"])
    val_data = collect_logits(model, val_loader, device, use_bf16)
    test_data = collect_logits(model, test_loader, device, use_bf16)

    temp = fit_temperature(val_data["logits"], val_data["labels"])
    val_probs = 1.0 / (1.0 + np.exp(-(val_data["logits"] / temp)))
    thr = find_high_precision_threshold(
        val_probs,
        val_data["labels"],
        target_precision=float(cfg["evaluation"]["target_precision"]),
        f_beta=float(cfg["evaluation"]["f_beta"]),
    )

    val_report = compute_detailed_metrics(val_data, temp, thr)
    test_report = compute_detailed_metrics(test_data, temp, thr)
    test_report["backbone"] = ckpt["backbone"]
    test_report["num_params"] = int(sum(p.numel() for p in model.parameters()))

    calib = {
        "temperature": temp,
        "threshold": thr,
        "val_summary": {k: val_report[k] for k in ("precision", "recall", "f1", "f0.5", "pr_auc", "roc_auc")},
    }
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "test_report.json").write_text(json.dumps(test_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== Calibration (from Val) ===")
    print(f"Temperature: {temp:.4f} | High-Precision Threshold: {thr:.4f}")
    print(f"\n=== Test Set Report ({len(test_ds)} images) ===")
    print(
        f"PR-AUC={test_report['pr_auc']:.4f} | ROC-AUC={test_report['roc_auc']:.4f} | "
        f"Precision={test_report['precision']:.4f} | Recall={test_report['recall']:.4f} | "
        f"F1={test_report['f1']:.4f} | F0.5={test_report['f0.5']:.4f}"
    )
    print(f"Confusion Matrix: {test_report['confusion_matrix']}")
    print("Per-Category Breakdown:")
    for cat, stats in test_report["by_category"].items():
        print(f"  - {cat}: {stats}")
    if test_report["top_false_positives"]:
        print("False Positives:")
        for item in test_report["top_false_positives"]:
            print(f"  FP ({item['prob']:.3f}) [{item['category']}/{item['subcategory']}]: {item['path']}")
    if test_report["top_false_negatives"]:
        print("False Negatives:")
        for item in test_report["top_false_negatives"]:
            print(f"  FN ({item['prob']:.3f}) [{item['category']}/{item['subcategory']}]: {item['path']}")


if __name__ == "__main__":
    main()
