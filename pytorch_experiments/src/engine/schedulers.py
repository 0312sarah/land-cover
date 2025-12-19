from __future__ import annotations

from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def build_scheduler(optim: Optimizer, sched_cfg: Optional[dict]):
    """
    Supported:
      - none (default)
      - step: {step_size: int, gamma: float}
      - cosine: {t_max: int, eta_min: float}
      - plateau: {mode: min/max, factor: float, patience: int, min_lr: float}
    """
    if not sched_cfg:
        return None, "none"

    name = str(sched_cfg.get("name", "none")).lower()
    if name == "none":
        return None, "none"

    if name == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return StepLR(optim, step_size=step_size, gamma=gamma), "step"

    if name == "cosine":
        t_max = int(sched_cfg.get("t_max", 50))
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        return CosineAnnealingLR(optim, T_max=t_max, eta_min=eta_min), "cosine"

    if name == "plateau":
        mode = str(sched_cfg.get("mode", "min")).lower()
        factor = float(sched_cfg.get("factor", 0.5))
        patience = int(sched_cfg.get("patience", 5))
        min_lr = float(sched_cfg.get("min_lr", 0.0))
        return ReduceLROnPlateau(
            optim,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        ), "plateau"

    raise ValueError(f"Unknown scheduler: {name}")
