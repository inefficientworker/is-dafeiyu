"""Observe a real optimizer step through online distillation on augmented images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from is_dafeiyu.dataset import DafeiyuDataset, build_transforms
from is_dafeiyu.model import (
    create_classifier,
    freeze_backbone,
    frozen_state_digest,
    resolve_device,
)
from is_dafeiyu.train import seed_everything, train_one_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(7)
    torch.set_num_threads(4)
    checkpoint = torch.load(
        args.run / "model/best.pt", map_location="cpu", weights_only=False
    )
    device = resolve_device()
    student, _ = create_classifier(
        checkpoint["backbone"], pretrained=False, img_size=224
    )
    teacher, _ = create_classifier(
        checkpoint["backbone"], pretrained=False, img_size=224
    )
    for model in (student, teacher):
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    freeze_backbone(student)
    teacher.eval().requires_grad_(False)
    mean, std = tuple(checkpoint["mean"]), tuple(checkpoint["std"])
    dataset = DafeiyuDataset(
        args.run / "splits/train.csv",
        build_transforms(224, mean, std, True),
        is_train=True,
    )
    # Include several labels/subcategories, including real transparent images.
    indices = []
    seen = set()
    for i, row in enumerate(dataset.rows):
        if row["subcategory"] not in seen:
            seen.add(row["subcategory"])
            indices.append(i)
    indices += [i for i, r in enumerate(dataset.rows) if r["label"] == 1][:8]
    loader = DataLoader(Subset(dataset, indices), batch_size=16, num_workers=0)
    captured = {"student": [], "teacher": []}
    handles = [
        model.register_forward_pre_hook(
            lambda model, inputs, key=key: captured[key].append(
                inputs[0].detach().float().cpu()
            )
        )
        for model, key in ((student, "student"), (teacher, "teacher"))
    ]
    before_frozen = frozen_state_digest(student)
    before_teacher = frozen_state_digest(teacher)
    head_before = [
        p.detach().cpu().clone() for p in student.get_classifier().parameters()
    ]
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=1e-3
    )
    normalization = tuple(
        torch.tensor(v, device=device).reshape(1, 3, 1, 1)
        for v in (mean, std, mean, std)
    )
    loss = train_one_epoch(
        student,
        loader,
        optimizer,
        torch.nn.BCEWithLogitsLoss(reduction="none"),
        device,
        True,
        0.02,
        kd_alpha=0.5,
        teacher=teacher,
        teacher_normalization=normalization,
    )
    for handle in handles:
        handle.remove()
    delta = max(
        float((s - t).abs().max())
        for s, t in zip(captured["student"], captured["teacher"])
    )
    assert delta < 1e-6 and len(captured["student"]) == len(captured["teacher"])
    assert frozen_state_digest(student) == before_frozen
    assert frozen_state_digest(teacher) == before_teacher
    assert any(
        not torch.equal(before, after.detach().cpu())
        for before, after in zip(head_before, student.get_classifier().parameters())
    )
    report = {
        "passed": True,
        "images": len(indices),
        "batches": len(captured["student"]),
        "loss": loss,
        "max_input_delta": delta,
        "frozen_student_unchanged": True,
        "teacher_unchanged": True,
        "student_head_updated": True,
    }
    (args.run / "aligned_distillation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
