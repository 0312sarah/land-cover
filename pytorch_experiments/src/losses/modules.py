from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, weight: Optional[Sequence[float]] = None, ignore_index: Optional[int] = None):
        super().__init__()
        kwargs = {}
        if weight is not None:
            kwargs["weight"] = torch.tensor(weight, dtype=torch.float32)
        if ignore_index is not None:
            kwargs["ignore_index"] = int(ignore_index)
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def to(self, device):
        self.loss = self.loss.to(device)
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Multi-class dice over all classes, ignoring `ignore_index` if set.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets = targets.long()

        valid = torch.ones_like(targets, dtype=torch.bool)
        if self.ignore_index is not None:
            valid = targets != self.ignore_index

        targets_clamped = torch.clamp(targets, min=0)
        one_hot = F.one_hot(targets_clamped, num_classes=num_classes).float()  # (B,H,W,C)
        one_hot = one_hot * valid.unsqueeze(-1)

        probs = probs.permute(0, 2, 3, 1)  # (B,H,W,C)
        probs = probs * valid.unsqueeze(-1)

        dims = (0, 1, 2)
        intersection = torch.sum(probs * one_hot, dims)
        denom = torch.sum(probs, dims) + torch.sum(one_hot, dims)
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class GlobalKLLoss(nn.Module):
    """
    KL(GT_hist || pred_hist) computed per-sample, excluding specified classes (default [0,1]).
    Aligns with leaderboard metric: compare class distributions (histogram) per image.
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-6,
        exclude_classes: Optional[Iterable[int]] = None,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        base_excl = {0, 1}
        if exclude_classes:
            base_excl.update(int(c) for c in exclude_classes)
        self.exclude_classes: List[int] = sorted(base_excl)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B,C,H,W)
        B, C, H, W = probs.shape
        device = logits.device

        kl_terms = []
        for b in range(B):
            target_b = targets[b]
            valid = torch.ones_like(target_b, dtype=torch.bool)
            if self.ignore_index is not None:
                valid = target_b != self.ignore_index

            target_valid = target_b[valid]
            if target_valid.numel() == 0:
                continue

            tgt_hist = torch.bincount(
                torch.clamp(target_valid, min=0).view(-1),
                minlength=self.num_classes,
            ).float()
            pred_sum = probs[b].reshape(C, -1).sum(dim=1)

            if self.exclude_classes:
                excl = torch.tensor(self.exclude_classes, device=device, dtype=torch.long)
                tgt_hist[excl] = 0.0
                pred_sum[excl] = 0.0

            tgt_total = tgt_hist.sum()
            pred_total = pred_sum.sum()
            if tgt_total <= 0 or pred_total <= 0:
                continue

            tgt_dist = tgt_hist / (tgt_total + self.eps)
            pred_dist = pred_sum / (pred_total + self.eps)

            kl = (tgt_dist * (torch.log(tgt_dist + self.eps) - torch.log(pred_dist + self.eps))).sum()
            kl_terms.append(kl)

        if not kl_terms:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(kl_terms).mean()


class WeightedSumLoss(nn.Module):
    def __init__(self, losses: Sequence[nn.Module], weights: Sequence[float]):
        super().__init__()
        if len(losses) != len(weights):
            raise ValueError("losses and weights must have same length")
        self.losses = nn.ModuleList(losses)
        self.weights = [float(w) for w in weights]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for w, loss_fn in zip(self.weights, self.losses):
            total = total + w * loss_fn(logits, targets)
        return total
