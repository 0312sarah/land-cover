from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

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

