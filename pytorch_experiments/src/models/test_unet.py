from pathlib import Path
import torch

from src.data.loader import build_dataloader
from src.models.build import build_model


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    ds_root = repo_root / "pytorch_experiments" / "dataset"

    dl = build_dataloader(root=ds_root, split="train", batch_size=2, num_workers=0, shuffle=False)
    images, masks = next(iter(dl))

    cfg = {
        "model": {
            "name": "unet",
            "in_channels": 4,
            "num_classes": 10,   # mets le bon nombre de classes du masque chez toi
            "num_layers": 4,
            "base_channels": 64,
            "upconv_filters": 96,
        }
    }

    model = build_model(cfg["model"])
    with torch.no_grad():
        out = model(images)

    print("in :", images.shape, images.dtype)
    print("out:", out.shape, out.dtype)  # (B, num_classes, H, W)
