"""
Data loading and preprocessing to mirror the TensorFlow baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tifffile import TiffFile


class LandCoverData:
    """Dataset constants shared across training and inference."""

    IMG_SIZE = 256
    N_CHANNELS = 4
    N_CLASSES = 10
    IGNORED_CLASSES_IDX = [0, 1]

    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356

    CLASSES = [
        "no_data",
        "clouds",
        "artificial",
        "cultivated",
        "broadleaf",
        "coniferous",
        "herbaceous",
        "natural",
        "snow",
        "water",
    ]

    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )


def _read_tif(path: Path) -> np.ndarray:
    with TiffFile(path) as tif:
        arr = tif.asarray()
    return arr


def _augment_tf_like(image: torch.Tensor, mask: Optional[torch.Tensor], aug_cfg: "Augmentations") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Replicate TF baseline stochastic flips/rotations."""
    if aug_cfg.hflip and torch.rand(()) > 0.5:
        image = torch.flip(image, dims=[2])
        if mask is not None:
            mask = torch.flip(mask, dims=[1])
    if aug_cfg.vflip and torch.rand(()) > 0.5:
        image = torch.flip(image, dims=[1])
        if mask is not None:
            mask = torch.flip(mask, dims=[0])
    if aug_cfg.rotate:
        r = torch.rand(())
        if r > 0.5:
            image = torch.rot90(image, k=1, dims=(1, 2))
            if mask is not None:
                mask = torch.rot90(mask, k=1, dims=(0, 1))
        elif torch.rand(()) > 0.5:
            image = torch.rot90(image, k=3, dims=(1, 2))
            if mask is not None:
                mask = torch.rot90(mask, k=3, dims=(0, 1))
    return image, mask


@dataclass
class Augmentations:
    hflip: bool = True
    vflip: bool = True
    rotate: bool = True


class LandCoverDataset(Dataset):
    def __init__(
        self,
        image_files: Sequence[Path],
        mode: str,
        augmentations: Optional[Augmentations] = None,
    ) -> None:
        assert mode in ("train", "val", "test")
        self.image_files: List[Path] = list(image_files)
        self.mode = mode
        self.augmentations = augmentations

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_pair(self, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        image = _read_tif(image_path)  # (H,W,C), uint16
        mask_path = image_path.parent.parent / "masks" / image_path.name
        mask = _read_tif(mask_path)  # (H,W), uint8
        image_t = torch.from_numpy(image).float().permute(2, 0, 1)  # C,H,W
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = _read_tif(image_path)
        return torch.from_numpy(image).float().permute(2, 0, 1)

    def __getitem__(self, idx: int):
        image_path = self.image_files[idx]
        sample_id = int(image_path.stem)
        if self.mode in ("train", "val"):
            image_t, mask_t = self._load_pair(image_path)
        else:
            image_t, mask_t = self._load_image(image_path), None

        if self.mode == "train" and self.augmentations is not None:
            image_t, mask_t = _augment_tf_like(image_t, mask_t, self.augmentations)

        image_t = image_t / float(LandCoverData.TRAIN_PIXELS_MAX)
        if mask_t is None:
            return image_t, sample_id
        return image_t, mask_t, sample_id


def build_dataloader(
    files: Sequence[Path],
    mode: str,
    batch_size: int,
    num_workers: int,
    augmentations: Optional[Augmentations],
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    dataset = LandCoverDataset(files, mode=mode, augmentations=augmentations)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )

