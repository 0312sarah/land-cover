"""
Inference / submission generation.
Command: python -m baseline.infer --config configs/infer/infer.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.config import load_config
from baseline.data import LandCoverData, build_dataloader, Augmentations
from baseline.metrics import distributions_from_logits
from baseline.model import UNet
from baseline.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch baseline inference")
    parser.add_argument("--config", "-c", required=True, type=str, help="YAML config path")
    return parser.parse_args()


def resolve_xp_dir(cfg) -> Path:
    xp_rootdir = Path(cfg.inference.xp_rootdir).expanduser().resolve()
    assert xp_rootdir.is_dir(), f"xp_rootdir not found: {xp_rootdir}"
    if cfg.inference.xp_name == "last":
        dirs = [d for d in xp_rootdir.iterdir() if d.is_dir()]
        xp_dir = Path(max(str(d) for d in dirs))
    else:
        xp_dir = xp_rootdir / cfg.inference.xp_name
    assert xp_dir.is_dir(), f"xp_dir not found: {xp_dir}"
    return xp_dir


def build_file_lists(cfg, xp_dir: Path, dataset_folder: Path) -> Tuple[List[Path], int]:
    if cfg.inference.set == "test":
        files = sorted(dataset_folder.glob("test/images/*.tif"))
    else:
        val_samples = pd.read_csv(xp_dir / "val_samples.csv", squeeze=True)
        val_files = [dataset_folder / f"train/images/{int(i)}.tif" for i in val_samples]
        if cfg.inference.set == "val":
            files = val_files
        else:
            val_set = set(val_files)
            files = [f for f in sorted(dataset_folder.glob("train/images/*.tif")) if f not in val_set]
    return files, len(files)


def sanity_checks(df: pd.DataFrame, class_cols: List[str]) -> None:
    assert list(df.columns) == ["sample_id"] + class_cols, "Unexpected columns in submission"
    assert df.shape[1] == 1 + len(class_cols), "Incorrect number of columns"
    assert np.isfinite(df[class_cols].values).all(), "Non-finite values in predictions"
    assert (df[class_cols].values >= 0).all(), "Negative values in predictions"
    row_sums = df[class_cols].sum(axis=1).values
    if not np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5):
        raise ValueError("Row sums are not close to 1")
    print("Example predictions:")
    print(df.head())
    print("Min per column:", df[class_cols].min().to_dict())
    print("Max per column:", df[class_cols].max().to_dict())


@torch.no_grad()
def run_inference(model: UNet, loader: DataLoader, device: torch.device, eps: float = 1e-8):
    model.eval()
    all_dists = []
    all_ids = []
    for batch in tqdm(loader, total=len(loader)):
        images, sample_ids = batch
        sample_ids = [int(s) for s in sample_ids]
        images = images.to(device)
        logits = model(images)
        dist = distributions_from_logits(logits, exclude_classes=[0, 1], eps=eps)
        all_dists.append(dist.cpu())
        all_ids.extend(sample_ids)
    return torch.cat(all_dists, dim=0), all_ids


def main():
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())

    set_seed(cfg.seed if hasattr(cfg, "seed") else None)
    device = get_device()

    dataset_folder = Path(cfg.data.root).expanduser().resolve()
    assert dataset_folder.is_dir(), f"dataset_folder not found: {dataset_folder}"

    xp_dir = resolve_xp_dir(cfg)
    files, dataset_size = build_file_lists(cfg, xp_dir, dataset_folder)

    batch_size = cfg.inference.batch_size
    num_workers = getattr(cfg.inference, "num_workers", 4)
    loader = build_dataloader(
        files,
        mode="test",
        batch_size=batch_size,
        num_workers=num_workers,
        augmentations=Augmentations(False, False, False),
        shuffle=False,
        drop_last=False,
    )

    model = UNet(
        in_channels=LandCoverData.N_CHANNELS,
        num_classes=LandCoverData.N_CLASSES,
        num_layers=cfg.model.num_layers,
        base_filters=getattr(cfg.model, "base_filters", 64),
        upconv_filters=getattr(cfg.model, "upconv_filters", 96),
    ).to(device)

    checkpoint_epoch = getattr(cfg.inference, "checkpoint_epoch", None)
    checkpoint_name = getattr(cfg.inference, "checkpoint", None)
    if checkpoint_name is not None and str(checkpoint_name).lower() == "best":
        ckpt_path = xp_dir / "checkpoints" / "best.pt"
        ckpt_label = "best"
    elif checkpoint_epoch is not None:
        ckpt_path = xp_dir / "checkpoints" / f"epoch{checkpoint_epoch}.pt"
        ckpt_label = f"epoch{checkpoint_epoch}"
    else:
        raise ValueError("Provide inference.checkpoint_epoch or inference.checkpoint ('best').")

    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])

    dists, sample_ids = run_inference(model, loader, device=device, eps=getattr(cfg.inference, "eps", 1e-8))
    columns = [
        "artificial",
        "cultivated",
        "broadleaf",
        "coniferous",
        "herbaceous",
        "natural_material",
        "snow",
        "water",
    ]
    df = pd.DataFrame(dists.numpy(), index=pd.Index(sample_ids, name="sample_id"), columns=columns)
    df.insert(0, "sample_id", df.index)
    sanity_checks(df.reset_index(drop=True), columns)

    out_csv = getattr(cfg.inference, "output_csv", None)
    if out_csv is None:
        out_csv = xp_dir / f"{ckpt_label}_{cfg.inference.set}_predicted.csv"
    out_csv = Path(out_csv).expanduser().resolve()
    print(f"Saving prediction CSV to {out_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
