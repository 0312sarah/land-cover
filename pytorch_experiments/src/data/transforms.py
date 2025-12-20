from __future__ import annotations

import random
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch

Tensor = torch.Tensor
ImageMask = Tuple[Tensor, Optional[Tensor]]


class Compose:
    """Apply a list of transforms on (image, mask)."""

    def __init__(self, transforms: Iterable[Callable[[Tensor, Optional[Tensor]], ImageMask]]):
        self.transforms = list(transforms)

    def __call__(self, image: Tensor, mask: Optional[Tensor] = None) -> ImageMask:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Normalize:
    """Per-channel normalization."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.registered = False
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: Tensor, mask: Optional[Tensor] = None) -> ImageMask:
        mean = self.mean.to(image.device, non_blocking=True)
        std = self.std.to(image.device, non_blocking=True)
        image = (image - mean) / std
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Tensor, mask: Optional[Tensor] = None) -> ImageMask:
        if random.random() < self.p:
            image = image.flip(-1)
            if mask is not None:
                mask = mask.flip(-1)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Tensor, mask: Optional[Tensor] = None) -> ImageMask:
        if random.random() < self.p:
            image = image.flip(-2)
            if mask is not None:
                mask = mask.flip(-2)
        return image, mask


class RandomRotate90:
    """Rotate by k*90 degrees."""

    def __init__(self):
        pass

    def __call__(self, image: Tensor, mask: Optional[Tensor] = None) -> ImageMask:
        k = random.randint(0, 3)
        if k:
            image = torch.rot90(image, k, dims=(-2, -1))
            if mask is not None:
                mask = torch.rot90(mask, k, dims=(-2, -1))
        return image, mask


class RandomCrop:
    """Random crop to (h, w)."""

    def __init__(self, size: Sequence[int]):
        if len(size) != 2:
            raise ValueError("Crop size must be a sequence of length 2: (height, width)")
        self.th, self.tw = int(size[0]), int(size[1])

    def __call__(self, image: Tensor, mask: Optional[Tensor] = None) -> ImageMask:
        _, h, w = image.shape
        if h < self.th or w < self.tw:
            raise ValueError(f"Crop size {(self.th, self.tw)} exceeds image size {(h, w)}")

        top = random.randint(0, h - self.th)
        left = random.randint(0, w - self.tw)

        image = image[:, top : top + self.th, left : left + self.tw]
        if mask is not None:
            mask = mask[top : top + self.th, left : left + self.tw]
        return image, mask


def build_transforms(aug_cfg: Optional[dict], split: str) -> Optional[Callable[[Tensor, Optional[Tensor]], ImageMask]]:
    """
    Build transforms for the given split based on config.
    Expected schema:
    augmentation:
      train:
        normalize: {mean: [...], std: [...]}
        geometric:
          random_hflip: true/false
          random_vflip: true/false
          random_rotate90: true/false
          random_crop: [H, W]  # optional
        photometric:
          brightness: 0.0
          contrast: 0.0
          gamma: 0.0
      val:
        normalize: {mean: [...], std: [...]}
    """
    aug_cfg = aug_cfg or {}
    split_cfg = aug_cfg.get(split, {}) or {}

    ops = []

    geom = split_cfg.get("geometric", {}) or {}
    if geom.get("random_hflip", False):
        ops.append(RandomHorizontalFlip())
    if geom.get("random_vflip", False):
        ops.append(RandomVerticalFlip())
    if geom.get("random_rotate90", False):
        ops.append(RandomRotate90())
    if "random_crop" in geom and geom["random_crop"]:
        ops.append(RandomCrop(geom["random_crop"]))

    norm = split_cfg.get("normalize")
    if norm and "mean" in norm and "std" in norm:
        ops.append(Normalize(norm["mean"], norm["std"]))

    if not ops:
        return None
    return Compose(ops)
