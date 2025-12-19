from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import torch


def update_confusion(conf: np.ndarray, preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Update confusion matrix in-place.
    preds: (B, H, W) long
    targets: (B, H, W) long
    """
    with torch.no_grad():
        preds = preds.view(-1).to(torch.int64)
        targets = targets.view(-1).to(torch.int64)
        valid = (targets >= 0) & (targets < num_classes)
        if not torch.any(valid):
            return conf
        preds = preds[valid]
        targets = targets[valid]
        idx = targets * num_classes + preds
        bincount = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += bincount.view(num_classes, num_classes).cpu().numpy()
    return conf


def confusion_to_metrics(conf: np.ndarray) -> Dict[str, Dict]:
    """
    Returns per-class and mean metrics: precision, recall, iou.
    conf: (C, C) np.ndarray (rows = true, cols = pred)
    """
    eps = 1e-7
    tp = np.diag(conf)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom_iou = tp + fp + fn

    iou = tp / (denom_iou + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    metrics = {
        "per_class": {
            "iou": iou.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
        "mean": {
            "miou": float(np.mean(iou)),
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
        },
        "confusion": conf.astype(int).tolist(),
    }
    return metrics


def save_confusion(conf: np.ndarray, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conf.tolist(), f)
