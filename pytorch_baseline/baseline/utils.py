"""
Misc utilities: seeding, logging, filesystem helpers.
"""
from __future__ import annotations

import os
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(root: Path, name: Optional[str] = None) -> Path:
    if name is None:
        name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def save_config_copy(config_path: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / "config.yaml"
    with open(config_path, "r") as src, dest.open("w") as dst:
        dst.write(src.read())


def save_namespace(config_ns: Namespace, path: Path) -> None:
    """Persist the Namespace as YAML for reproducibility."""
    def _to_obj(obj):
        if isinstance(obj, Namespace):
            return {k: _to_obj(v) for k, v in vars(obj).items()}
        if isinstance(obj, (list, tuple)):
            return [_to_obj(x) for x in obj]
        return obj
    with path.open("w") as f:
        yaml.safe_dump(_to_obj(config_ns), f)


def init_log_file(log_path: Path) -> None:
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            f.write("epoch,train_loss,val_loss,val_kl\n")


def append_log(log_path: Path, epoch: int, train_loss: float, val_loss: float, val_kl: float) -> None:
    with log_path.open("a") as f:
        f.write(f"{epoch},{train_loss},{val_loss},{val_kl}\n")


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

