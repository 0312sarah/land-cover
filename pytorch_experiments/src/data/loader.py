from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import LandCoverDataset


def build_dataloader(
        root : str | Path, 
        split : str,
        batch_size : int = 32,
        num_workers : int = 0,
        shuffle: bool = False,
        transform=None,
        sampler=None,
):
    ds = LandCoverDataset(root=root, split=split, transform=transform)
    effective_shuffle = shuffle and sampler is None

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=effective_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    if num_workers > 0:
        dl_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    dl = DataLoader(ds, **dl_kwargs)
    return dl


def build_train_val_loaders(
        root: str | Path,
        batch_size: int = 32,
        num_workers: int = 0,
        val_ratio: float = 0.2,
        seed: int = 42,
        train_transform=None,
        val_transform=None,
        sampling_cfg: Optional[dict] = None,
):
    """
    Split the train set into train/val subsets with a deterministic seed.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    images_dir = Path(root) / "train" / "images"
    image_files = sorted(images_dir.glob("*.tif"))
    n_total = len(image_files)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    if n_train <= 0:
        raise ValueError(f"val_ratio={val_ratio} leaves no samples for training (n_total={n_total})")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    val_indices = perm[:n_val].tolist()
    train_indices = perm[n_val:].tolist()

    train_files = [image_files[i] for i in train_indices]
    val_files = [image_files[i] for i in val_indices]

    train_ds = LandCoverDataset(root=root, split="train", transform=train_transform, files=train_files)
    val_ds = LandCoverDataset(root=root, split="train", transform=val_transform, files=val_files)

    sampler = _build_sampler(train_ds, sampling_cfg)
    effective_shuffle = sampler is None

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        common_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, val_loader


def _build_sampler(dataset: LandCoverDataset, sampling_cfg: Optional[dict]):
    if not sampling_cfg:
        return None
    strategy = (sampling_cfg.get("strategy") or "none").lower()
    if strategy == "none":
        return None

    if strategy == "presence":
        class_weights = sampling_cfg.get("class_weights")
        min_weight = float(sampling_cfg.get("min_weight", 1.0))
        weights = _compute_presence_weights(
            dataset,
            class_weights=class_weights,
            min_weight=min_weight,
        )
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _compute_presence_weights(dataset: LandCoverDataset, class_weights: Optional[Iterable[float]], min_weight: float):
    """
    Oversample tiles containing rare classes.
    Weight per tile = min_weight + sum(class_weights[c] for each present class c).
    class_weights should be a list/iterable of length num_classes (ordered after mapping 0..C-1).
    """
    if dataset.mask_dir is None:
        raise ValueError("Sampling by presence only applies to training split with masks.")

    if class_weights is None:
        class_weights = [1.0] * dataset.num_classes
    class_weights = list(class_weights)
    if len(class_weights) != dataset.num_classes:
        raise ValueError(f"class_weights must have length {dataset.num_classes}, got {len(class_weights)}")

    weights = []
    for img_path in dataset.images_files:
        mask_path = dataset.mask_dir / img_path.name
        mask = tifffile.imread(mask_path)
        unique = np.unique(mask)
        w = min_weight
        for raw_cls in unique:
            mapped = dataset.label_map.get(int(raw_cls))
            if mapped is not None:
                w += class_weights[mapped]
        weights.append(float(w))
    return weights
