"""
Loss functions: CE, Dice, and GlobalKLHistogram combinations.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _gather_active_classes(num_classes: int, exclude_classes: Optional[List[int]]) -> torch.Tensor:
    exclude_classes = exclude_classes or []
    include = [c for c in range(num_classes) if c not in exclude_classes]
    return torch.tensor(include, dtype=torch.long)


def global_kl_histogram(
    logits: torch.Tensor,
    target: torch.Tensor,
    exclude_classes: Optional[List[int]] = None,
    eps: float = 1e-8,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable KL on per-image class histograms (excludes ignore/exclude classes)."""
    b, num_classes, h, w = logits.shape
    prob = torch.softmax(logits, dim=1)
    pred_dist = prob.mean(dim=(2, 3))  # (B, C)

    if ignore_index is not None:
        valid_mask = (target != ignore_index).float()
        denom = valid_mask.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        one_hot = F.one_hot(target.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid_mask.unsqueeze(1)
    else:
        denom = torch.tensor(h * w, device=logits.device, dtype=torch.float32).expand(b, 1, 1)
        one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    gt_dist = one_hot.sum(dim=(2, 3)) / denom.squeeze(-1)

    keep_idx = _gather_active_classes(num_classes, exclude_classes).to(logits.device)
    pred_dist = pred_dist.index_select(1, keep_idx)
    gt_dist = gt_dist.index_select(1, keep_idx)

    pred_dist = pred_dist / pred_dist.sum(dim=1, keepdim=True).clamp_min(eps)
    gt_dist = gt_dist / gt_dist.sum(dim=1, keepdim=True).clamp_min(eps)

    kl = (gt_dist + eps) * torch.log((gt_dist + eps) / (pred_dist + eps))
    return kl.sum(dim=1).mean()


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    exclude_classes: Optional[List[int]] = None,
    smooth: float = 1.0,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    prob = torch.softmax(logits, dim=1)

    if ignore_index is not None:
        valid_mask = (target != ignore_index).float()
        target = target.clone()
        target[valid_mask == 0] = 0
        prob = prob * valid_mask.unsqueeze(1)

    one_hot = F.one_hot(target.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()

    if exclude_classes:
        keep_idx = _gather_active_classes(num_classes, exclude_classes).to(logits.device)
        prob = prob.index_select(1, keep_idx)
        one_hot = one_hot.index_select(1, keep_idx)

    dims = (0, 2, 3)
    intersection = torch.sum(prob * one_hot, dims)
    cardinality = torch.sum(prob + one_hot, dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


class LossFactory:
    """Factory to build composite losses from a name and parameters."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def build(
        self,
        name: str,
        class_weights: Optional[List[float]] = None,
        ignore_index: Optional[int] = None,
        kl_weight: float = 1.0,
        dice_weight: float = 1.0,
        exclude_classes: Optional[List[int]] = None,
        eps: float = 1e-8,
        smooth: float = 1.0,
    ):
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        def ce_loss(logits, target):
            kwargs: Dict = {}
            if ignore_index is not None:
                kwargs["ignore_index"] = ignore_index
            weight = weight_tensor.to(logits.device) if weight_tensor is not None else None
            return F.cross_entropy(logits, target, weight=weight, **kwargs)

        def make_ignore_value():
            return ignore_index if ignore_index is not None else None

        def ce_only(logits, target):
            return ce_loss(logits, target)

        def ce_kl(logits, target):
            return ce_loss(logits, target) + kl_weight * global_kl_histogram(
                logits, target, exclude_classes=exclude_classes, eps=eps, ignore_index=make_ignore_value()
            )

        def ce_dice_kl(logits, target):
            dice = soft_dice_loss(
                logits, target, exclude_classes=exclude_classes, smooth=smooth, ignore_index=make_ignore_value()
            )
            kl = global_kl_histogram(
                logits, target, exclude_classes=exclude_classes, eps=eps, ignore_index=make_ignore_value()
            )
            return ce_loss(logits, target) + dice_weight * dice + kl_weight * kl

        name = name.lower()
        if name == "cross_entropy":
            return ce_only
        if name == "ce_global_kl":
            return ce_kl
        if name == "ce_dice_global_kl":
            return ce_dice_kl
        raise ValueError(f"Unknown loss name: {name}")
