from pathlib import Path
import torch

from src.data.loader import build_dataloader


if __name__ == "__main__":
    # Chemin robuste vers le dataset depuis le repo
    repo_root = Path(__file__).resolve().parents[3]
    ds_root = repo_root / "pytorch_experiments" / "dataset"

    dl = build_dataloader(
        root=ds_root,
        split="train",
        batch_size=4,
        num_workers=0,
        shuffle=True,
    )

    images, masks = next(iter(dl))

    print("images:", images.shape, images.dtype)  # (B, C, H, W) float32
    print("masks :", masks.shape, masks.dtype)    # (B, H, W) int64

    # sanity checks
    assert images.ndim == 4 and images.shape[1] == 4
    assert masks.ndim == 3
    assert images.dtype == torch.float32
    assert masks.dtype == torch.int64

    print("[OK] DataLoader batching works.")
