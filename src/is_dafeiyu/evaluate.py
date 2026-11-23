"""Calibrate on an independent holdout in FP32, then evaluate on Test."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.special import expit
from scipy.stats import beta as beta_distribution
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
from is_dafeiyu.provenance import (
    check_checkpoint_splits,
    configured_splits,
    file_sha256,
    split_provenance,
)


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> dict:
    model.eval()
    logits_list, labels_list, cats, subcats, paths, groups = [], [], [], [], [], []
    amp_enabled, amp_dtype = resolve_amp(device, use_amp)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(images).squeeze(-1)
        logits_list.append(logits.float().cpu().numpy())
        labels_list.append(batch["label"].numpy())
        cats.extend(batch["category"])
        subcats.extend(batch["subcategory"])
        paths.extend(batch["path"])
        groups.extend(batch["cluster_id"])
    return {
        "logits": np.concatenate(logits_list),
        "labels": np.concatenate(labels_list).astype(int),
        "categories": cats,
        "subcategories": subcats,
        "paths": paths,
        "groups": groups,
    }


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Find scalar temperature T > 0 minimizing NLL on validation set."""
    if not np.isfinite(val_logits).all():
        raise ValueError("Cannot calibrate non-finite logits")
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


DEFAULT_CONSTRAINTS = {
    "target_precision": 0.98,
    "min_recall": 0.8,
    "max_fpr": 0.01,
    "max_subcategory_fpr": 0.05,
}


def operating_constraints(config: dict | None = None) -> dict:
    config = config or {}
    result = {
        key: float(config.get(key, default))
        for key, default in DEFAULT_CONSTRAINTS.items()
    }
    if (
        any(not 0 <= value <= 1 for value in result.values())
        or result["min_recall"] <= 0
    ):
        raise ValueError("Operating constraints must be in [0, 1], with min_recall > 0")
    return result


def select_operating_point(
    scores: np.ndarray, labels: np.ndarray, subcategories: list[str], **constraints
) -> dict:
    """Maximize recall subject to explicit precision and per-negative-group FPR limits.

    Scores may be logits or deployed probabilities. Search actual score boundaries,
    preserving high-confidence distinctions and never silently relaxing constraints.
    """
    limits = operating_constraints(constraints)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if (
        not len(scores)
        or len(scores) != len(labels)
        or len(scores) != len(subcategories)
    ):
        raise ValueError(
            "Scores, labels and subcategories must have equal nonzero length"
        )
    if not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError(
            "Operating point selection needs finite scores and both labels"
        )
    positives = labels == 1
    negatives = ~positives
    category_masks = [
        negatives & (np.asarray(subcategories) == name)
        for name in sorted(set(subcategories))
    ]
    category_masks = [mask for mask in category_masks if mask.any()]
    best = None
    diagnostic = None
    values = np.unique(scores)
    for index, cutoff in enumerate(values):
        predictions = scores >= cutoff
        tp = int((predictions & positives).sum())
        fp = int((predictions & negatives).sum())
        recall = tp / int(positives.sum())
        precision = tp / max(tp + fp, 1)
        fpr = fp / int(negatives.sum())
        worst_fpr = max(float(predictions[mask].mean()) for mask in category_masks)
        threshold = float(cutoff)
        if index:
            midpoint = float(values[index - 1] + (cutoff - values[index - 1]) / 2)
            if values[index - 1] < midpoint <= cutoff:
                threshold = midpoint
        point = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "false_positives": fp,
            "fpr": fpr,
            "worst_subcategory_fpr": worst_fpr,
        }
        if recall >= limits["min_recall"]:
            if diagnostic is None or (precision, -worst_fpr, recall) > (
                diagnostic["precision"],
                -diagnostic["worst_subcategory_fpr"],
                diagnostic["recall"],
            ):
                diagnostic = point
            feasible = (
                precision >= limits["target_precision"]
                and fpr <= limits["max_fpr"]
                and worst_fpr <= limits["max_subcategory_fpr"]
            )
            improves = best is None or (recall, -fp, -worst_fpr) > (
                best["recall"],
                -best["false_positives"],
                -best["worst_subcategory_fpr"],
            )
            if feasible and improves:
                best = point
    return {
        "accepted": best is not None,
        "constraints": limits,
        "operating_point": best,
        "diagnostic_point": diagnostic,
    }


def find_high_precision_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.98,
    f_beta: float = 0.5,
    *,
    subcategories: list[str] | None = None,
    **constraints,
) -> float:
    # f_beta is retained for existing callers; it is no longer a fallback objective.
    result = select_operating_point(
        probs,
        labels,
        subcategories or ["all"] * len(labels),
        target_precision=target_precision,
        **constraints,
    )
    if not result["accepted"]:
        raise ValueError(f"No threshold meets operating constraints: {result}")
    return result["operating_point"]["threshold"]


def fpr_upper_bound(false_positives: int, count: int) -> float:
    """Exact one-sided 95% Clopper-Pearson bound, assuming independent trials."""
    return (
        1.0
        if false_positives == count
        else float(
            beta_distribution.ppf(0.95, false_positives + 1, count - false_positives)
        )
    )


def summarize_groups(
    labels: np.ndarray, preds: np.ndarray, groups: list[str]
) -> dict[str, dict]:
    by_group: dict[str, dict] = {}
    indices: dict[str, list[int]] = collections.defaultdict(list)
    for i, group in enumerate(groups):
        indices[group].append(i)

    for group, idxs in sorted(indices.items()):
        negative = labels[idxs] == 0
        group_preds = preds[idxs]
        n_neg = int(negative.sum())
        n_pos = len(idxs) - n_neg
        stats = {"count": len(idxs)}
        if n_neg:
            fp = int(group_preds[negative].sum())
            stats.update(
                false_positives=fp,
                negative_count=n_neg,
                fpr=fp / n_neg,
                fpr_upper_95=fpr_upper_bound(fp, n_neg),
            )
        if n_pos:
            fn = n_pos - int(group_preds[~negative].sum())
            stats.update(false_negatives=fn, recall=1.0 - fn / n_pos)
        by_group[group] = stats
    return by_group


def compute_detailed_metrics(data: dict, temp: float, threshold: float) -> dict:
    scores = np.asarray(data["logits"], dtype=np.float64) / temp
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model logits")
    probs = expit(scores)
    labels = data["labels"]
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

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
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "negative_fpr_upper_95": fpr_upper_bound(int(fp), int(tn + fp)),
        "max_abs_logit": float(np.max(np.abs(data["logits"]))),
        "samples": [
            {
                "path": path,
                "label": int(labels[i]),
                "logit": float(data["logits"][i]),
                "probability": float(probs[i]),
                "predicted": int(preds[i]),
            }
            for i, path in enumerate(data["paths"])
        ],
        "by_category": summarize_groups(labels, preds, data["categories"]),
        "by_subcategory": summarize_groups(labels, preds, data["subcategories"]),
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
    model, _ = create_classifier(
        backbone=ckpt["backbone"], pretrained=False, img_size=ckpt["image_size"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    eval_tf = build_transforms(
        ckpt["image_size"], tuple(ckpt["mean"]), tuple(ckpt["std"]), is_train=False
    )
    provenance = split_provenance(configured_splits(cfg))
    check_checkpoint_splits(ckpt, provenance)
    val_ds = DafeiyuDataset(cfg["data"]["calibration_csv"], eval_tf, is_train=False)
    test_ds = DafeiyuDataset(cfg["data"]["test_csv"], eval_tf, is_train=False)

    num_workers = int(cfg["data"].get("num_workers", 4))
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    # Training AMP must never determine the calibration/evaluation precision.
    val_data = collect_logits(model, val_loader, device, False)
    test_data = collect_logits(model, test_loader, device, False)

    temp = fit_temperature(val_data["logits"], val_data["labels"])
    val_probs = expit(val_data["logits"].astype(np.float64) / temp)
    selection = select_operating_point(
        val_probs,
        val_data["labels"],
        val_data["subcategories"],
        **operating_constraints(cfg.get("evaluation")),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    attempt = {
        **selection,
        "temperature": temp,
        "precision_backend": "pytorch_fp32",
        "checkpoint_sha256": file_sha256(ckpt_path),
        "data_provenance": provenance,
    }
    (out_dir / "calibration_attempt.json").write_text(
        json.dumps(attempt, indent=2), encoding="utf-8"
    )
    if not selection["accepted"]:
        raise ValueError(
            "Calibration did not meet constraints; see calibration_attempt.json"
        )
    thr = selection["operating_point"]["threshold"]

    val_report = compute_detailed_metrics(val_data, temp, thr)
    test_report = compute_detailed_metrics(test_data, temp, thr)
    test_report["backbone"] = ckpt["backbone"]
    test_report["num_params"] = int(sum(p.numel() for p in model.parameters()))

    calib = {
        **attempt,
        "temperature": temp,
        "threshold": thr,
        "calibration_summary": {
            k: val_report[k]
            for k in ("precision", "recall", "f1", "f0.5", "pr_auc", "roc_auc")
        },
        "calibration_by_subcategory": val_report["by_subcategory"],
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "test_report.json").write_text(
        json.dumps(test_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== Calibration (independent calibration split, FP32) ===")
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
            print(
                f"  FP ({item['prob']:.3f}) [{item['category']}/{item['subcategory']}]: {item['path']}"
            )
    if test_report["top_false_negatives"]:
        print("False Negatives:")
        for item in test_report["top_false_negatives"]:
            print(
                f"  FN ({item['prob']:.3f}) [{item['category']}/{item['subcategory']}]: {item['path']}"
            )


if __name__ == "__main__":
    main()
