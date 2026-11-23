"""Two-stage fine-tuning script for Dafeiyu classification on Intel XPU."""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from is_dafeiyu.dataset import DafeiyuDataset, build_transforms
from is_dafeiyu.model import (
    create_classifier,
    freeze_backbone,
    resolve_amp,
    resolve_device,
    unfreeze_last_blocks,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def load_teacher_logits(path: str | Path) -> dict[str, float]:
    """Load precomputed teacher logits keyed by sample path from .json or .npz."""
    p = Path(path)
    if p.suffix.lower() == ".npz":
        data = np.load(p, allow_pickle=False)
        return {str(k): float(v) for k, v in zip(data["paths"].tolist(), data["logits"].tolist())}
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw = payload.get("logits", payload)
    return {str(k): float(v) for k, v in raw.items()}


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_cats: list[str] = []

    amp_enabled, amp_dtype = resolve_amp(device, use_amp)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images).squeeze(-1)
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_cats.extend(batch["category"])

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= 0.5).astype(int)

    pr_auc = float(average_precision_score(labels_np, probs))
    roc_auc = float(roc_auc_score(labels_np, probs))
    f1 = float(f1_score(labels_np, preds, zero_division=0))
    prec = float(precision_score(labels_np, preds, zero_division=0))
    rec = float(recall_score(labels_np, preds, zero_division=0))

    cat_fpr: dict[str, float] = {}
    by_cat: dict[str, list[int]] = collections.defaultdict(list)
    for cat, lbl, pred in zip(all_cats, labels_np, preds):
        if lbl == 0:
            by_cat[cat].append(int(pred))
    for cat, neg_preds in sorted(by_cat.items()):
        cat_fpr[cat] = float(sum(neg_preds) / max(1, len(neg_preds)))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "cat_fpr_at_0.5": cat_fpr,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    label_smoothing: float,
    scaler: torch.amp.GradScaler | None = None,
    kd_alpha: float = 0.0,
    kd_temperature: float = 3.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    amp_enabled, amp_dtype = resolve_amp(device, use_amp)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.float32, non_blocking=True)
        if label_smoothing > 0.0:
            targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            targets = labels

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images).squeeze(-1)
        logits_f32 = logits.float()
        loss_hard = criterion(logits_f32, targets)
        if kd_alpha > 0.0 and "teacher_logit" in batch:
            t_logits = batch["teacher_logit"].to(device, dtype=torch.float32, non_blocking=True)
            t_probs = torch.sigmoid(t_logits / kd_temperature)
            loss_kd = F.binary_cross_entropy_with_logits(logits_f32 / kd_temperature, t_probs) * (
                kd_temperature * kd_temperature
            )
            loss = (1.0 - kd_alpha) * loss_hard + kd_alpha * loss_kd
        else:
            loss = loss_hard

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(cfg["seed"])

    device = resolve_device(cfg["training"]["device"])
    use_amp = bool(cfg["training"].get("use_bf16", True))
    amp_enabled, amp_dtype = resolve_amp(device, use_amp)
    print(f"[train] Using device: {device} (amp={amp_enabled}, dtype={amp_dtype})")
    if device.type == "xpu":
        print(f"[train] XPU device name: {torch.xpu.get_device_name(0)}")
    elif device.type == "cuda":
        print(f"[train] CUDA device name: {torch.cuda.get_device_name(0)}")

    model, data_cfg = create_classifier(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        drop_rate=cfg["model"]["drop_rate"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
        img_size=cfg["data"]["image_size"],
    )
    mean = tuple(data_cfg["mean"])
    std = tuple(data_cfg["std"])
    image_size = cfg["data"]["image_size"]

    distill_cfg = cfg.get("distillation") or {}
    teacher_logits = None
    kd_alpha = 0.0
    kd_temp = 3.0
    if distill_cfg.get("teacher_logits_path"):
        teacher_logits = load_teacher_logits(distill_cfg["teacher_logits_path"])
        kd_alpha = float(distill_cfg.get("alpha", 0.5))
        kd_temp = float(distill_cfg.get("temperature", 3.0))
        print(f"[train] Distillation enabled: {len(teacher_logits)} teacher logits, alpha={kd_alpha}, T={kd_temp}")

    train_tf = build_transforms(image_size, mean, std, is_train=True)
    val_tf = build_transforms(image_size, mean, std, is_train=False)

    train_ds = DafeiyuDataset(cfg["data"]["train_csv"], train_tf, is_train=True, teacher_logits=teacher_logits)
    val_ds = DafeiyuDataset(cfg["data"]["val_csv"], val_tf, is_train=False)

    n_pos = sum(r["label"] for r in train_ds.rows)
    n_neg = len(train_ds.rows) - n_pos
    # Use sqrt-dampened pos_weight so precision remains high while keeping strong recall
    pos_weight_val = float(np.sqrt(n_neg / max(1, n_pos)) * cfg["training"]["pos_weight_scale"])
    print(f"[train] Train samples: {len(train_ds)} (pos={n_pos}, neg={n_neg}), pos_weight={pos_weight_val:.2f}")
    print(f"[train] Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=False,
    )

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    label_smoothing = float(cfg["training"]["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and device.type == "cuda" and amp_dtype == torch.float16))

    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    best_score = -1.0
    history: list[dict] = []
    t0 = time.time()

    def save_if_best(stage: str, epoch: int, train_loss: float, metrics: dict) -> None:
        nonlocal best_score
        # Selection metric: prioritize PR-AUC (primary) + F1
        score = metrics["pr_auc"]
        entry = {"stage": stage, "epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(entry)
        fpr_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics["cat_fpr_at_0.5"].items())
        print(
            f"[{stage} ep {epoch:02d}] loss={train_loss:.4f} "
            f"val_pr_auc={metrics['pr_auc']:.4f} roc_auc={metrics['roc_auc']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} | FPR: {fpr_str}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": cfg["model"]["backbone"],
                    "image_size": image_size,
                    "mean": mean,
                    "std": std,
                    "val_metrics": metrics,
                    "config": cfg,
                },
                best_path,
            )

    # --- Stage 1: Linear Probe ---
    print("\n=== Stage 1: Linear Probe (Backbone Frozen) ===")
    freeze_backbone(model)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["stage1_lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    for ep in range(1, int(cfg["training"]["stage1_epochs"]) + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            opt1,
            criterion,
            device,
            use_amp,
            label_smoothing,
            scaler=scaler,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temp,
        )
        metrics = evaluate_loader(model, val_loader, device, use_amp)
        save_if_best("stage1", ep, loss, metrics)

    # --- Stage 2: Partial / Full Unfreeze ---
    unfreeze_n = int(cfg["training"]["unfreeze_last_blocks"])
    print(f"\n=== Stage 2: Unfreezing {'All Blocks' if unfreeze_n <= 0 else f'Last {unfreeze_n} Blocks'} ===")
    bb_params, head_params = unfreeze_last_blocks(model, n_blocks=unfreeze_n)
    s2_lr = float(cfg["training"]["stage2_lr"])
    opt2 = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": s2_lr},
            {"params": head_params, "lr": s2_lr * 3.0},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    s2_epochs = int(cfg["training"]["stage2_epochs"])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=s2_epochs, eta_min=s2_lr * 0.05)

    for ep in range(1, s2_epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            opt2,
            criterion,
            device,
            use_amp,
            label_smoothing,
            scaler=scaler,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temp,
        )
        sched2.step()
        metrics = evaluate_loader(model, val_loader, device, use_amp)
        save_if_best("stage2", ep, loss, metrics)

    elapsed = time.time() - t0
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] Training completed in {elapsed / 60:.2f} min. Best Val PR-AUC={best_score:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
