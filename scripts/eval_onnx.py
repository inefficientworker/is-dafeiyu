"""Evaluate and benchmark single-file ONNX models on test.csv using ONNX Runtime CPU."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
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

from is_dafeiyu.dataset import build_transforms, load_rgb_image
from is_dafeiyu.export import image_to_rgba_uint8


def evaluate_onnx(
    onnx_path: Path,
    test_csv: Path,
    threshold: float,
    image_size: int = 224,
    input_format: str = "rgba_uint8",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    warmup: int = 10,
) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rows = list(csv.DictReader(test_csv.open("r", encoding="utf-8")))
    eval_tf = build_transforms(image_size, mean, std, is_train=False)

    inputs: list[np.ndarray] = []
    labels: list[int] = []
    cats: list[str] = []
    paths: list[str] = []

    for r in rows:
        p = r["path"]
        if input_format == "rgba_uint8":
            arr = image_to_rgba_uint8(p, image_size)
        else:
            rgb = load_rgb_image(p, random_bg=False)
            arr = eval_tf(rgb).unsqueeze(0).numpy().astype(np.float32)
        inputs.append(arr)
        labels.append(int(r["label"]))
        cats.append(r["category"])
        paths.append(p)

    for arr in inputs[:warmup]:
        sess.run(None, {input_name: arr})

    probs: list[float] = []
    latencies_ms: list[float] = []
    for arr in inputs:
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: arr})[0]
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        probs.append(float(np.asarray(out).reshape(-1)[0]))

    probs_np = np.array(probs, dtype=np.float64)
    labels_np = np.array(labels, dtype=int)
    preds_np = (probs_np >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np, labels=[0, 1]).ravel()

    by_cat: dict[str, dict] = {}
    cat_indices: dict[str, list[int]] = collections.defaultdict(list)
    for i, c in enumerate(cats):
        cat_indices[c].append(i)
    for cat, idxs in sorted(cat_indices.items()):
        c_labels = labels_np[idxs]
        c_preds = preds_np[idxs]
        if (c_labels == 0).all():
            c_fp = int((c_preds == 1).sum())
            by_cat[cat] = {"count": len(idxs), "false_positives": c_fp, "fpr": float(c_fp / len(idxs))}
        else:
            c_fn = int((c_preds == 0).sum())
            by_cat[cat] = {"count": len(idxs), "false_negatives": c_fn, "recall": float(1.0 - c_fn / len(idxs))}

    return {
        "onnx_path": str(onnx_path),
        "size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 3),
        "threshold": float(threshold),
        "latency_mean_ms": round(float(np.mean(latencies_ms)), 3),
        "latency_median_ms": round(float(np.median(latencies_ms)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "precision": float(precision_score(labels_np, preds_np, zero_division=0)),
        "recall": float(recall_score(labels_np, preds_np, zero_division=0)),
        "f1": float(f1_score(labels_np, preds_np, zero_division=0)),
        "f0.5": float(fbeta_score(labels_np, preds_np, beta=0.5, zero_division=0)),
        "pr_auc": float(average_precision_score(labels_np, probs_np)),
        "roc_auc": float(roc_auc_score(labels_np, probs_np)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "by_category": by_cat,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--test-csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--input-format", type=str, choices=["rgba_uint8", "nchw_float32"], default="rgba_uint8")
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    calib_path = Path(args.calibration) if args.calibration else onnx_path.parent / "calibration.json"
    thr = 0.5
    if calib_path.exists():
        thr = float(json.loads(calib_path.read_text(encoding="utf-8")).get("threshold", 0.5))
    if args.threshold is not None:
        thr = args.threshold

    res = evaluate_onnx(onnx_path, Path(args.test_csv), thr, input_format=args.input_format)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
