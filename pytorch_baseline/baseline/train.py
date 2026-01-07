"""
Training entrypoint: python -m baseline.train --config configs/train/unet_ce_global_kl.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import optim

from baseline.config import load_config
from baseline.data import LandCoverData, Augmentations, build_dataloader
from baseline.losses import LossFactory
from baseline.metrics import batch_val_kl
from baseline.model import UNet
from baseline.utils import append_log, get_device, init_log_file, make_run_dir, save_config_copy, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch baseline training")
    parser.add_argument("--config", "-c", required=True, type=str, help="YAML config path")
    return parser.parse_args()


def _compute_class_weights(mode: Optional[str]) -> Optional[List[float]]:
    if mode is None:
        return None
    if mode == "inverse_frequency":
        class_weight = (1 / LandCoverData.TRAIN_CLASS_COUNTS[2:]) * LandCoverData.TRAIN_CLASS_COUNTS[2:].sum() / (
            LandCoverData.N_CLASSES - 2
        )
        full = np.zeros(LandCoverData.N_CLASSES, dtype=np.float32)
        full[2:] = class_weight
        return full.tolist()
    raise ValueError(f"Unknown class weight mode: {mode}")


def _split_train_val(dataset_folder: Path, val_samples_csv: Optional[Path], seed: Optional[int]):
    train_files = list(dataset_folder.glob("train/images/*.tif"))
    train_files = random.sample(train_files, len(train_files))
    devset_size = len(train_files)
    if val_samples_csv is not None:
        val_ids = pd.read_csv(val_samples_csv, squeeze=True)
        val_files = [dataset_folder / f"train/images/{int(i)}.tif" for i in val_ids]
        train_files = [f for f in train_files if f not in set(val_files)]
        valset_size = len(val_files)
        trainset_size = len(train_files)
        assert valset_size + trainset_size == devset_size
    else:
        valset_size = int(len(train_files) * 0.1)
        train_files, val_files = train_files[valset_size:], train_files[:valset_size]
        # mirror TF baseline bookkeeping
        trainset_size = len(train_files) - valset_size
    return train_files, val_files, trainset_size, valset_size


def train_one_epoch(model, loader, loss_fn, optimizer, device, max_steps: int):
    model.train()
    total_loss = 0.0
    total_samples = 0
    steps = 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = images.shape[0]
        total_loss += loss.item() * bs
        total_samples += bs
        steps += 1
        if steps >= max_steps:
            break
    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss


@torch.no_grad()
def validate(model, loader, loss_fn, device, max_steps: int):
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    total_samples = 0
    steps = 0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        kl = batch_val_kl(logits, masks)
        bs = images.shape[0]
        total_loss += loss.item() * bs
        total_kl += kl.item() * bs
        total_samples += bs
        steps += 1
        if steps >= max_steps:
            break
    mean_loss = total_loss / max(total_samples, 1)
    mean_kl = total_kl / max(total_samples, 1)
    return mean_loss, mean_kl


@torch.no_grad()
def compute_precision_recall(
    model: UNet,
    files,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    ignore_index: Optional[int],
    exclude_classes: Optional[List[int]],
) -> Dict[int, Dict[str, float]]:
    include_classes = [c for c in range(LandCoverData.N_CLASSES) if not exclude_classes or c not in exclude_classes]
    loader = build_dataloader(
        files,
        mode="val",
        batch_size=batch_size,
        num_workers=num_workers,
        augmentations=None,
        shuffle=False,
        drop_last=False,
    )

    tp = torch.zeros(LandCoverData.N_CLASSES, dtype=torch.long)
    pred_pos = torch.zeros(LandCoverData.N_CLASSES, dtype=torch.long)
    true_pos = torch.zeros(LandCoverData.N_CLASSES, dtype=torch.long)

    model.eval()
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        if ignore_index is not None:
            valid = masks != ignore_index
        else:
            valid = torch.ones_like(masks, dtype=torch.bool)
        for c in include_classes:
            pred_c = (preds == c) & valid
            true_c = (masks == c) & valid
            tp[c] += (pred_c & true_c).sum().item()
            pred_pos[c] += pred_c.sum().item()
            true_pos[c] += true_c.sum().item()

    metrics: Dict[int, Dict[str, float]] = {}
    for c in include_classes:
        precision = float(tp[c] / pred_pos[c]) if pred_pos[c] > 0 else 0.0
        recall = float(tp[c] / true_pos[c]) if true_pos[c] > 0 else 0.0
        metrics[c] = {"precision": precision, "recall": recall}
    return metrics


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)

    set_seed(cfg.seed if hasattr(cfg, "seed") else None)
    device = get_device()

    dataset_folder = Path(cfg.data.root).expanduser().resolve()
    xp_rootdir = Path(cfg.training.xp_rootdir).expanduser().resolve()
    assert dataset_folder.is_dir(), f"dataset_folder not found: {dataset_folder}"
    assert xp_rootdir.is_dir(), f"xp_rootdir not found: {xp_rootdir}"

    val_samples_csv = None
    if getattr(cfg.data, "val_samples_csv", None):
        val_samples_csv = Path(cfg.data.val_samples_csv).expanduser().resolve()
    train_files, val_files, trainset_size, valset_size = _split_train_val(dataset_folder, val_samples_csv, cfg.seed)

    run_dir = make_run_dir(xp_rootdir, getattr(cfg.training, "xp_name", None))
    save_config_copy(config_path, run_dir)
    pd.Series([int(f.stem) for f in val_files], name="sample_id", dtype="uint32").to_csv(run_dir / "val_samples.csv", index=False)
    log_path = run_dir / "log.txt"
    init_log_file(log_path)

    aug_cfg = Augmentations(
        hflip=getattr(cfg.augmentations, "hflip", True),
        vflip=getattr(cfg.augmentations, "vflip", True),
        rotate=getattr(cfg.augmentations, "rotate", True),
    )

    batch_size = cfg.data.batch_size
    num_workers = getattr(cfg.data, "num_workers", 4)
    train_loader = build_dataloader(
        train_files, mode="train", batch_size=batch_size, num_workers=num_workers, augmentations=aug_cfg, shuffle=True, drop_last=True
    )
    val_loader = build_dataloader(
        val_files, mode="val", batch_size=batch_size, num_workers=num_workers, augmentations=None, shuffle=False, drop_last=True
    )

    model = UNet(
        in_channels=LandCoverData.N_CHANNELS,
        num_classes=LandCoverData.N_CLASSES,
        num_layers=cfg.model.num_layers,
        base_filters=getattr(cfg.model, "base_filters", 64),
        upconv_filters=getattr(cfg.model, "upconv_filters", 96),
    ).to(device)

    class_weights = getattr(cfg.loss, "class_weights", None)
    if isinstance(class_weights, str):
        class_weights = _compute_class_weights(class_weights)
    loss_factory = LossFactory(num_classes=LandCoverData.N_CLASSES)
    loss_fn = loss_factory.build(
        name=cfg.loss.name,
        class_weights=class_weights,
        ignore_index=getattr(cfg.loss, "ignore_index", None),
        kl_weight=getattr(cfg.loss, "kl_weight", 1.0),
        dice_weight=getattr(cfg.loss, "dice_weight", 1.0),
        exclude_classes=getattr(cfg.loss, "exclude_classes", [0, 1]),
        eps=getattr(cfg.loss, "eps", 1e-8),
        smooth=getattr(cfg.loss, "smooth", 1.0),
    )

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=getattr(cfg.training, "weight_decay", 0.0))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=getattr(cfg.training, "lr_patience", 20),
        factor=getattr(cfg.training, "lr_factor", 0.5),
        verbose=True,
    )

    epochs = cfg.training.epochs
    train_steps = max(1, trainset_size // batch_size)
    val_steps = max(1, valset_size // batch_size)

    best_val_kl = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, max_steps=train_steps)
        val_loss, val_kl = validate(model, val_loader, loss_fn, device, max_steps=val_steps)
        scheduler.step(val_loss)

        ckpt_path = run_dir / "checkpoints" / f"epoch{epoch}.pt"
        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
        append_log(log_path, epoch, train_loss, val_loss, val_kl)
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_kl={val_kl:.6f}")
        if val_kl < best_val_kl:
            best_val_kl = val_kl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        best_ckpt = run_dir / "checkpoints" / "best.pt"
        torch.save({"model_state": best_state, "best_val_kl": best_val_kl}, best_ckpt)
        model.load_state_dict(best_state)

        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        class_metrics = compute_precision_recall(
            model,
            val_files,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            ignore_index=getattr(cfg.loss, "ignore_index", None),
            exclude_classes=getattr(cfg.loss, "exclude_classes", [0, 1]),
        )
        metrics_path = metrics_dir / "metrics.txt"
        with metrics_path.open("w") as f:
            f.write(f"best_val_kl: {best_val_kl}\n")
            f.write("class,precision,recall\n")
            for c, stats in class_metrics.items():
                name = LandCoverData.CLASSES[c]
                f.write(f"{name},{stats['precision']},{stats['recall']}\n")
        print(f"Saved best checkpoint and metrics to {metrics_path}")


if __name__ == "__main__":
    main()
