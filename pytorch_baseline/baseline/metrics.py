"""
Metrics including the challenge KL divergence.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _drop_and_normalize(dist: torch.Tensor, exclude_classes: List[int], eps: float) -> torch.Tensor:
    keep = [i for i in range(dist.shape[1]) if i not in exclude_classes]
    keep_idx = torch.tensor(keep, device=dist.device, dtype=torch.long)
    dist = dist.index_select(1, keep_idx)
    dist = dist / dist.sum(dim=1, keepdim=True).clamp_min(eps)
    return dist


def kl_divergence(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return ((y_true + eps) * torch.log((y_true + eps) / (y_pred + eps))).sum(dim=1)


def batch_val_kl(
    logits: torch.Tensor,
    target: torch.Tensor,
    exclude_classes: Optional[List[int]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the validation KL metric per batch (mean over samples).
    - exclude classes 0 and 1
    - renormalize remaining classes
    """
    if exclude_classes is None:
        exclude_classes = [0, 1]
    num_classes = logits.shape[1]
    prob = torch.softmax(logits, dim=1)
    pred_dist = prob.mean(dim=(2, 3))  # (B, C)

    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    gt_dist = one_hot.sum(dim=(2, 3))
    pred_dist = _drop_and_normalize(pred_dist, exclude_classes, eps)
    gt_dist = _drop_and_normalize(gt_dist, exclude_classes, eps)

    kl = kl_divergence(gt_dist, pred_dist, eps=eps)
    return kl.mean()


def distributions_from_logits(
    logits: torch.Tensor, exclude_classes: Optional[List[int]] = None, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute per-sample class distributions from logits for inference.
    - softmax probabilities
    - average spatially
    - remove excluded classes and renormalize
    """
    if exclude_classes is None:
        exclude_classes = [0, 1]
    prob = torch.softmax(logits, dim=1)
    dist = prob.mean(dim=(2, 3))
    dist = _drop_and_normalize(dist, exclude_classes, eps)
    return dist

