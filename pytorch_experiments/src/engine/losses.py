from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def build_loss(loss_cfg: Optional[dict], num_classes: Optional[int] = None, device: Optional[torch.device] = None) -> nn.Module:
    """
    Supported losses:
      - cross_entropy (default): {ignore_index, weight}
    """
    loss_cfg = loss_cfg or {}
    name = str(loss_cfg.get("name", "cross_entropy")).lower()

    if name == "cross_entropy":
        weight = loss_cfg.get("weight")
        if weight is not None:
            if num_classes is not None and len(weight) != num_classes:
                raise ValueError(f"weight length must match num_classes ({num_classes}), got {len(weight)}")
            weight = torch.tensor(weight, dtype=torch.float32)
            if device is not None:
                weight = weight.to(device)
        ignore_index = loss_cfg.get("ignore_index")
        if ignore_index is not None:
            ignore_index = int(ignore_index)
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        if device is not None:
            criterion = criterion.to(device)
        return criterion

    raise ValueError(f"Unknown loss: {name}")
