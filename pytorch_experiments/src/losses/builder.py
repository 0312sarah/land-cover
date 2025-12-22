from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .modules import CrossEntropyLossWrapper, DiceLoss, GlobalKLLoss, WeightedSumLoss


def _ensure_weight_tensor(weight: Optional[Sequence[float]], num_classes: Optional[int], device: torch.device):
    if weight is None:
        return None
    if num_classes is not None and len(weight) != num_classes:
        raise ValueError(f"weight length must match num_classes ({num_classes}), got {len(weight)}")
    t = torch.tensor(weight, dtype=torch.float32, device=device)
    return t


def build_loss(loss_cfg: Optional[dict], num_classes: Optional[int] = None, device: Optional[torch.device] = None) -> nn.Module:
    """
    Supported names:
      - cross_entropy
      - ce_global_kl
      - ce_dice_global_kl
    """
    device = device or torch.device("cpu")
    loss_cfg = loss_cfg or {}
    name = str(loss_cfg.get("name", "cross_entropy")).lower()

    weight = _ensure_weight_tensor(loss_cfg.get("weight"), num_classes, device)
    ignore_index = loss_cfg.get("ignore_index")
    exclude_classes_cfg = loss_cfg.get("exclude_classes") or []
    exclude_classes = sorted({0, 1, *exclude_classes_cfg})
    dice_weight = float(loss_cfg.get("dice_weight", 1.0))
    kl_weight = float(loss_cfg.get("kl_weight", 1.0))
    smooth = float(loss_cfg.get("smooth", 1e-6))
    eps = float(loss_cfg.get("eps", 1e-6))

    def ce_loss():
        ce = CrossEntropyLossWrapper(weight=weight, ignore_index=ignore_index)
        return ce.to(device)

    def kl_loss():
        if num_classes is None:
            raise ValueError("num_classes is required for global KL loss")
        return GlobalKLLoss(
            num_classes=num_classes,
            eps=eps,
            exclude_classes=exclude_classes,
            ignore_index=ignore_index,
        ).to(device)

    if name == "cross_entropy":
        return ce_loss()
    if name == "ce_global_kl":
        return WeightedSumLoss(
            losses=[ce_loss(), kl_loss()],
            weights=[1.0, kl_weight],
        )
    if name == "ce_dice_global_kl":
        return WeightedSumLoss(
            losses=[ce_loss(), DiceLoss(smooth=smooth, ignore_index=ignore_index).to(device), kl_loss()],
            weights=[1.0, dice_weight, kl_weight],
        )

    raise ValueError(f"Unknown loss: {name}")
