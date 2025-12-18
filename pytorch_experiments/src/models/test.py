import torch
from pathlib import Path
from src.data.loader import build_dataloader

repo_root = Path(__file__).resolve().parents[3]
ds_root = repo_root / "pytorch_experiments" / "dataset"
dl = build_dataloader(ds_root, "train", batch_size=2, num_workers=0, shuffle=False)
_, masks = next(iter(dl))
print("min/max:", masks.min().item(), masks.max().item())
print("unique (first batch):", torch.unique(masks)[:20])
