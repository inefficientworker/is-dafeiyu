"""Model wrapper and Intel XPU device utilities."""

from __future__ import annotations

import hashlib

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


def resolve_amp(device: torch.device, use_amp: bool = True) -> tuple[bool, torch.dtype]:
    """Resolve whether AMP is enabled and which dtype (bfloat16 vs float16) the device natively supports."""
    if not use_amp or device.type not in ("xpu", "cuda"):
        return False, torch.float32
    if device.type == "xpu":
        return True, torch.bfloat16
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


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
    try:
        model = timm.create_model(backbone, **kwargs)
    except TypeError as exc:
        if "img_size" in kwargs and "img_size" in str(exc):
            kwargs.pop("img_size")
            model = timm.create_model(backbone, **kwargs)
        else:
            raise
    data_cfg = timm.data.resolve_model_data_config(model)
    return model, data_cfg


def _get_head_params(model: nn.Module) -> list[nn.Parameter]:
    head = model.get_classifier()
    if isinstance(head, nn.Module):
        return list(head.parameters())
    # Fallback by parameter name
    return [
        p
        for n, p in model.named_parameters()
        if any(k in n for k in ("head", "fc", "classifier"))
    ]


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone weights, leaving only the classification head trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in _get_head_params(model):
        p.requires_grad = True


def set_training_mode(model: nn.Module) -> None:
    """Train unfrozen subtrees while keeping frozen features and buffers deterministic."""
    model.eval()

    def enable(module: nn.Module) -> None:
        parameters = list(module.parameters())
        if parameters and all(p.requires_grad for p in parameters):
            module.train()
        else:
            for child in module.children():
                enable(child)

    enable(model)


def frozen_state_digest(model: nn.Module) -> str:
    """Fingerprint frozen parameters and buffers for stage-level training audits."""
    state = {name: p for name, p in model.named_parameters() if not p.requires_grad}
    for name, buffer in model.named_buffers():
        parent = model.get_submodule(name.rpartition(".")[0])
        if not any(p.requires_grad for p in parent.parameters()):
            state[name] = buffer
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def unfreeze_last_blocks(
    model: nn.Module, n_blocks: int = 3
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Unfreeze the classification head, final norm/conv_head, and last n_blocks/stages of the backbone.

    If n_blocks <= 0, all backbone parameters are unfrozen.
    Returns (backbone_params, head_params) for differential learning rates.
    """
    head_ids = {id(p) for p in _get_head_params(model)}
    if n_blocks <= 0:
        for p in model.parameters():
            p.requires_grad = True
    else:
        freeze_backbone(model)

        # Unfreeze final norm / head-projection layers (ViT norm, MobileNet conv_head/norm_head, ConvNeXt head.norm)
        for tail_attr in (
            "norm",
            "fc_norm",
            "norm_pre",
            "conv_head",
            "norm_head",
            "head",
        ):
            tail_mod = getattr(model, tail_attr, None)
            if isinstance(tail_mod, nn.Module):
                for p in tail_mod.parameters():
                    p.requires_grad = True

        # Unfreeze last n transformer/mobilenet blocks or ConvNeXt stages
        if hasattr(model, "blocks"):
            blocks = list(model.blocks)
            for blk in blocks[-n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        elif hasattr(model, "stages"):
            stages = list(model.stages)
            for stg in stages[-n_blocks:]:
                for p in stg.parameters():
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_ids
    ]
    head_params = [
        p for p in model.parameters() if p.requires_grad and id(p) in head_ids
    ]
    return backbone_params, head_params
