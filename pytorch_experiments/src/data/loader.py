from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch

from .dataset import LandCoverDataset


def build_dataloader(
        root : str | Path, 
        split : str,
        batch_size : int = 32,
        num_workers : int = 0,
        shuffle: bool = False,
):
    ds = LandCoverDataset(root= root, split = split)
    dl = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dl


def build_train_val_loaders(
        root: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        val_ratio: float = 0.2,
        seed: int = 42,
):
    """
    Split the train set into train/val subsets with a deterministic seed.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    full_ds = LandCoverDataset(root=root, split="train")
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(f"val_ratio={val_ratio} leaves no samples for training (n_total={n_total})")

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
