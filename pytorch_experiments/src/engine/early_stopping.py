from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    start_epoch: int = 1


class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best: Optional[float] = None
        self.best_epoch: int = 0

    @classmethod
    def from_dict(cls, cfg_dict: dict | None) -> "EarlyStopping":
        cfg_dict = cfg_dict or {}
        cfg = EarlyStoppingConfig(
            enabled=bool(cfg_dict.get("enabled", False)),
            patience=int(cfg_dict.get("patience", 5)),
            min_delta=float(cfg_dict.get("min_delta", 0.0)),
            mode=str(cfg_dict.get("mode", "min")).lower(),
            start_epoch=int(cfg_dict.get("start_epoch", 1)),
        )
        if cfg.mode not in {"min", "max"}:
            raise ValueError(f"early_stop.mode must be 'min' or 'max', got {cfg.mode}")
        return cls(cfg)

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:
            return True
        delta = metric - self.best
        if self.cfg.mode == "min":
            return delta < -self.cfg.min_delta
        return delta > self.cfg.min_delta

    def step(self, metric: float, epoch: int) -> bool:
        """
        Returns True if training should stop.
        """
        if not self.cfg.enabled or epoch < self.cfg.start_epoch:
            # still track best
            if self.best is None or self._is_improvement(metric):
                self.best = metric
                self.best_epoch = epoch
            return False

        if self._is_improvement(metric):
            self.best = metric
            self.best_epoch = epoch
            return False

        if (epoch - self.best_epoch) >= self.cfg.patience:
            return True
        return False
