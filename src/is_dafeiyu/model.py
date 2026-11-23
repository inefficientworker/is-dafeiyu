"""Model wrapper and Intel XPU device utilities."""

from __future__ import annotations

import timm
import torch
from torch import nn


def resolve_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_cfg)


def create_classifier(
    backbone: str,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.15,
    img_size: int | None = None,
) -> tuple[nn.Module, dict]:
    kwargs: dict = {
        "pretrained": pretrained,
        "num_classes": 1,
        "drop_rate": drop_rate,
        "drop_path_rate": drop_path_rate,
    }
    if img_size is not None:
        kwargs["img_size"] = img_size
    model = timm.create_model(backbone, **kwargs)
    data_cfg = timm.data.resolve_model_data_config(model)
    return model, data_cfg


def _get_head_params(model: nn.Module) -> list[nn.Parameter]:
    head = model.get_classifier()
    if isinstance(head, nn.Module):
        return list(head.parameters())
    # Fallback by parameter name
    return [p for n, p in model.named_parameters() if any(k in n for k in ("head", "fc", "classifier"))]


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone weights, leaving only the classification head trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in _get_head_params(model):
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 3) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Unfreeze the classification head, final norm, and last n_blocks of the backbone.

    Returns (backbone_params, head_params) for differential learning rates.
    """
    freeze_backbone(model)
    head_ids = {id(p) for p in _get_head_params(model)}

    # Unfreeze final norm layers
    for norm_attr in ("norm", "fc_norm"):
        norm_mod = getattr(model, norm_attr, None)
        if isinstance(norm_mod, nn.Module):
            for p in norm_mod.parameters():
                p.requires_grad = True

    # Unfreeze last n transformer blocks (ViT/DINOv2/EVA) or last stage (ConvNeXt)
    if hasattr(model, "blocks"):
        blocks = list(model.blocks)
        for blk in blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    elif hasattr(model, "stages"):
        stages = list(model.stages)
        for stg in stages[-1:]:
            for p in stg.parameters():
                p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    return backbone_params, head_params
