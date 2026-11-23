"""Two-stage fine-tuning script for Dafeiyu classification on Intel XPU."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from is_dafeiyu.dataset import DafeiyuDataset, build_transforms
from is_dafeiyu.evaluate import (
    collect_logits,
    compute_detailed_metrics,
    operating_constraints,
    select_operating_point,
)
from is_dafeiyu.model import (
    create_classifier,
    freeze_backbone,
    frozen_state_digest,
    resolve_amp,
    resolve_device,
    set_training_mode,
    unfreeze_last_blocks,
)
from is_dafeiyu.provenance import (
    check_checkpoint_splits,
    configured_splits,
    split_provenance,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, device: torch.device, constraints: dict
) -> dict:
    data = collect_logits(model, loader, device, False)
    metrics = compute_detailed_metrics(data, 1.0, 0.5)
    return {
        key: metrics[key]
        for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "max_abs_logit")
    } | {
        "selection": select_operating_point(
            data["logits"], data["labels"], data["subcategories"], **constraints
        ),
        "by_subcategory_at_0.5": metrics["by_subcategory"],
        "evaluation_precision": "fp32",
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
    teacher: nn.Module | None = None,
    teacher_normalization: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
    subcategory_loss_weights: dict[str, float] | None = None,
) -> float:
    set_training_mode(model)
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
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(images).squeeze(-1)
        logits_f32 = logits.float()
        loss_hard = criterion(logits_f32, targets)
        if kd_alpha > 0.0 and teacher is not None:
            student_mean, student_std, teacher_mean, teacher_std = teacher_normalization
            # Use the exact same augmented pixels, including random alpha background.
            teacher_images = (
                images.float() * student_std + student_mean - teacher_mean
            ) / teacher_std
            with torch.no_grad():
                t_logits = teacher(teacher_images).squeeze(-1).float()
            if not torch.isfinite(t_logits).all():
                raise FloatingPointError("Non-finite teacher logits")
            t_probs = torch.sigmoid(t_logits / kd_temperature)
            loss_kd = F.binary_cross_entropy_with_logits(
                logits_f32 / kd_temperature, t_probs, reduction="none"
            ) * (kd_temperature * kd_temperature)
            blended = (1.0 - kd_alpha) * loss_hard + kd_alpha * loss_kd
            conflict = (t_logits >= 0) != (labels >= 0.5)
            loss = torch.where(conflict, loss_hard, blended)
        else:
            loss = loss_hard
        weights = loss.new_tensor(
            [(subcategory_loss_weights or {}).get(s, 1.0) for s in batch["subcategory"]]
        )
        loss = (loss * weights).sum() / weights.sum()
        if not torch.isfinite(logits_f32).all() or not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training output in batch: {batch['path']}"
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
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

    provenance = split_provenance(configured_splits(cfg))
    constraints = operating_constraints(cfg.get("evaluation"))
    out_dir = Path(cfg["training"]["output_dir"])
    if any(
        (out_dir / name).exists()
        for name in ("best.pt", "candidate.pt", "history.json")
    ):
        raise ValueError(
            f"Training output already exists; choose a fresh output_dir: {out_dir}"
        )

    distill_cfg = cfg.get("distillation") or {}
    if distill_cfg.get("teacher_logits_path"):
        raise ValueError(
            "Cached whole-image logits are incompatible with augmented training. Use distillation.teacher_checkpoint."
        )
    teacher = None
    teacher_normalization = None
    kd_alpha = float(distill_cfg.get("alpha", 0.0))
    kd_temp = float(distill_cfg.get("temperature", 3.0))
    if not 0 <= kd_alpha <= 1 or kd_temp <= 0:
        raise ValueError("Distillation requires alpha in [0,1] and temperature > 0")
    if kd_alpha > 0:
        teacher_ckpt = torch.load(
            distill_cfg["teacher_checkpoint"], map_location="cpu", weights_only=False
        )
        check_checkpoint_splits(teacher_ckpt, provenance)
        if teacher_ckpt["image_size"] != image_size:
            raise ValueError(
                "Teacher and student must use the same image size for aligned distillation"
            )
        teacher, _ = create_classifier(
            teacher_ckpt["backbone"], pretrained=False, img_size=image_size
        )
        teacher.load_state_dict(teacher_ckpt["model_state_dict"])
        teacher.to(device).eval().requires_grad_(False)
        teacher_normalization = tuple(
            torch.tensor(values, device=device).reshape(1, 3, 1, 1)
            for values in (mean, std, teacher_ckpt["mean"], teacher_ckpt["std"])
        )
        print(
            f"[train] Online aligned-view distillation, alpha={kd_alpha}, T={kd_temp}"
        )

    train_tf = build_transforms(image_size, mean, std, is_train=True)
    val_tf = build_transforms(image_size, mean, std, is_train=False)
    train_ds = DafeiyuDataset(cfg["data"]["train_csv"], train_tf, is_train=True)
    val_ds = DafeiyuDataset(cfg["data"]["val_csv"], val_tf, is_train=False)
    n_pos = sum(r["label"] for r in train_ds.rows)
    if "pos_weight_scale" in cfg["training"]:
        raise ValueError(
            "Use explicit training.pos_weight (default 1); automatic class-ratio weighting was removed"
        )
    pos_weight_val = float(cfg["training"].get("pos_weight", 1.0))
    loss_weights = cfg["training"].get("subcategory_loss_weights", {})
    if pos_weight_val <= 0 or any(float(w) <= 0 for w in loss_weights.values()):
        raise ValueError("Loss weights must be positive")
    print(
        f"[train] Train={len(train_ds)} pos={n_pos} neg={len(train_ds) - n_pos}, pos_weight={pos_weight_val:.2f}; val={len(val_ds)}"
    )
    # Equalize source-group contribution instead of counting each animation frame as independent.
    group_counts = Counter(
        r.get("cluster_id") or r.get("group_id") or r["path"] for r in train_ds.rows
    )
    sampler = WeightedRandomSampler(
        [
            1.0 / group_counts[r.get("cluster_id") or r.get("group_id") or r["path"]]
            for r in train_ds.rows
        ],
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=False,
    )

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=device), reduction="none"
    )
    label_smoothing = float(cfg["training"]["label_smoothing"])
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(amp_enabled and device.type == "cuda" and amp_dtype == torch.float16),
    )

    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    best_score = None
    accepted_checkpoint = False
    history: list[dict] = []
    training_audit: dict = {}
    t0 = time.time()

    def save_if_best(stage: str, epoch: int, train_loss: float, metrics: dict) -> None:
        nonlocal best_score, accepted_checkpoint
        selection = metrics["selection"]
        point = selection["operating_point"] or selection["diagnostic_point"]
        accepted = selection["accepted"]
        score = (
            accepted,
            point["recall"] if accepted else point["precision"],
            -point["worst_subcategory_fpr"],
            -point["false_positives"],
            metrics["pr_auc"],
        )
        entry = {"stage": stage, "epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(entry)
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"[{stage} ep {epoch:02d}] loss={train_loss:.4f} val_pr_auc={metrics['pr_auc']:.4f} "
            f"eligible={accepted} P={point['precision']:.4f} R={point['recall']:.4f} "
            f"worst_subcategory_FPR={point['worst_subcategory_fpr']:.4f}",
            flush=True,
        )
        if best_score is None or score > best_score:
            best_score = score
            accepted_checkpoint = accepted
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": cfg["model"]["backbone"],
                    "image_size": image_size,
                    "mean": mean,
                    "std": std,
                    "val_metrics": metrics,
                    "config": cfg,
                    "data_provenance": provenance,
                },
                best_path if accepted else out_dir / "candidate.pt",
            )

    # --- Stage 1: Linear Probe ---
    print("\n=== Stage 1: Linear Probe (Backbone Frozen) ===")
    freeze_backbone(model)
    stage1_frozen = frozen_state_digest(model)
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
            teacher=teacher,
            teacher_normalization=teacher_normalization,
            subcategory_loss_weights=loss_weights,
        )
        metrics = evaluate_loader(model, val_loader, device, constraints)
        save_if_best("stage1", ep, loss, metrics)

    training_audit["stage1"] = {
        "frozen_before": stage1_frozen,
        "frozen_after": frozen_state_digest(model),
    }
    (out_dir / "training_audit.json").write_text(json.dumps(training_audit, indent=2))
    if (
        training_audit["stage1"]["frozen_before"]
        != training_audit["stage1"]["frozen_after"]
    ):
        raise RuntimeError(
            "Frozen backbone parameters or buffers changed during stage 1"
        )

    # --- Stage 2: Partial / Full Unfreeze ---
    unfreeze_n = int(cfg["training"]["unfreeze_last_blocks"])
    print(
        f"\n=== Stage 2: Unfreezing {'All Blocks' if unfreeze_n <= 0 else f'Last {unfreeze_n} Blocks'} ==="
    )
    bb_params, head_params = unfreeze_last_blocks(model, n_blocks=unfreeze_n)
    stage2_frozen = frozen_state_digest(model)
    s2_lr = float(cfg["training"]["stage2_lr"])
    opt2 = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": s2_lr},
            {"params": head_params, "lr": s2_lr * 3.0},
        ],
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    s2_epochs = int(cfg["training"]["stage2_epochs"])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=s2_epochs, eta_min=s2_lr * 0.05
    )

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
            teacher=teacher,
            teacher_normalization=teacher_normalization,
            subcategory_loss_weights=loss_weights,
        )
        sched2.step()
        metrics = evaluate_loader(model, val_loader, device, constraints)
        save_if_best("stage2", ep, loss, metrics)

    training_audit["stage2"] = {
        "frozen_before": stage2_frozen,
        "frozen_after": frozen_state_digest(model),
    }
    (out_dir / "training_audit.json").write_text(json.dumps(training_audit, indent=2))
    if (
        training_audit["stage2"]["frozen_before"]
        != training_audit["stage2"]["frozen_after"]
    ):
        raise RuntimeError(
            "Frozen backbone parameters or buffers changed during stage 2"
        )

    elapsed = time.time() - t0
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not accepted_checkpoint:
        raise RuntimeError(
            f"No checkpoint met validation constraints; diagnostic candidate and history saved in {out_dir}"
        )
    print(f"\n[done] Training completed in {elapsed / 60:.2f} min -> {best_path}")


if __name__ == "__main__":
    main()
