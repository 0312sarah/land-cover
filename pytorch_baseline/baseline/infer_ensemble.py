"""
Deep ensemble inference / submission generation.
Command: python -m baseline.infer_ensemble --config configs/infer_ensemble/infer.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.config import load_config
from baseline.data import LandCoverData, build_dataloader, Augmentations
from baseline.infer import build_file_lists, resolve_xp_dir, sanity_checks
from baseline.metrics import distributions_from_logits, kl_divergence
from baseline.model import UNet
from baseline.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch baseline ensemble inference")
    parser.add_argument("--config", "-c", required=True, type=str, help="YAML config path")
    return parser.parse_args()


def build_model(cfg, device: torch.device) -> UNet:
    return UNet(
        in_channels=LandCoverData.N_CHANNELS,
        num_classes=LandCoverData.N_CLASSES,
        num_layers=cfg.model.num_layers,
        base_filters=getattr(cfg.model, "base_filters", 64),
        upconv_filters=getattr(cfg.model, "upconv_filters", 96),
    ).to(device)


def resolve_checkpoints(checkpoints: Sequence[str], xp_dir: Path) -> List[Path]:
    if len(checkpoints) == 0:
        raise ValueError("Provide at least one checkpoint path in inference.checkpoints.")
    resolved = []
    for ckpt in checkpoints:
        ckpt_path = Path(ckpt).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = (xp_dir / ckpt_path).resolve()
        else:
            ckpt_path = ckpt_path.resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resolved.append(ckpt_path)
    return resolved


def _identity(img: torch.Tensor) -> torch.Tensor:
    return img


def _hflip(img: torch.Tensor) -> torch.Tensor:
    return torch.flip(img, dims=[3])


def _vflip(img: torch.Tensor) -> torch.Tensor:
    return torch.flip(img, dims=[2])


def _rot90(img: torch.Tensor) -> torch.Tensor:
    return torch.rot90(img, k=1, dims=(2, 3))


def _rot180(img: torch.Tensor) -> torch.Tensor:
    return torch.rot90(img, k=2, dims=(2, 3))


def _rot270(img: torch.Tensor) -> torch.Tensor:
    return torch.rot90(img, k=3, dims=(2, 3))


def build_tta_transforms(tta_cfg: Sequence[str]) -> List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    if not tta_cfg:
        tta_cfg = ["none"]
    transforms = []
    for name in tta_cfg:
        key = str(name).lower()
        if key in ("none", "identity"):
            transforms.append((key, _identity))
        elif key == "hflip":
            transforms.append((key, _hflip))
        elif key == "vflip":
            transforms.append((key, _vflip))
        elif key == "rot90":
            transforms.append((key, _rot90))
        elif key == "rot180":
            transforms.append((key, _rot180))
        elif key == "rot270":
            transforms.append((key, _rot270))
        else:
            raise ValueError(f"Unknown TTA transform: {name}")
    return transforms


@torch.no_grad()
def predict_distributions_with_tta(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    tta: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]],
    eps: float,
    snow_floor: Optional[float] = None,
) -> Tuple[torch.Tensor, List[int], Optional[torch.Tensor], List[Tuple[str, float]]]:
    model.eval()
    tta_outputs = []
    sample_ids = None
    gt_reference: Optional[torch.Tensor] = None
    tta_kls: List[Tuple[str, float]] = []

    for name, transform in tta:
        dists = []
        ids = []
        gt_batches = []
        for batch in tqdm(loader, total=len(loader), desc=f"TTA:{name}"):
            if len(batch) == 3:
                images, masks, batch_ids = batch
            else:
                images, batch_ids = batch
                masks = None

            images = transform(images).to(device)
            logits = model(images)
            dist = distributions_from_logits(logits, exclude_classes=[0, 1], eps=eps, snow_floor=snow_floor)
            dists.append(dist.cpu())
            ids.extend(int(s) for s in batch_ids)

            if masks is not None and gt_reference is None:
                # Ground-truth distribution per sample (drop classes 0/1 + renorm).
                num_classes = LandCoverData.N_CLASSES
                one_hot = torch.nn.functional.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
                gt_dist = one_hot.sum(dim=(2, 3))
                gt_dist = gt_dist[:, 2:]
                gt_dist = gt_dist / gt_dist.sum(dim=1, keepdim=True).clamp_min(eps)
                gt_batches.append(gt_dist.cpu())

        current = torch.cat(dists, dim=0)
        if gt_reference is None and gt_batches:
            gt_reference = torch.cat(gt_batches, dim=0)

        if sample_ids is None:
            sample_ids = ids
        elif sample_ids != ids:
            raise RuntimeError("Sample IDs differ between TTA passes; unexpected dataloader ordering change.")
        tta_outputs.append(current)

        if gt_reference is not None:
            kl_val = kl_divergence(gt_reference, current, eps=eps).mean().item()
            tta_kls.append((name, kl_val))

    stacked = torch.stack(tta_outputs, dim=0)
    return stacked.mean(dim=0), sample_ids, gt_reference, tta_kls


def main():
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())

    set_seed(cfg.seed if hasattr(cfg, "seed") else None)
    device = get_device()

    dataset_folder = Path(cfg.data.root).expanduser().resolve()
    assert dataset_folder.is_dir(), f"dataset_folder not found: {dataset_folder}"

    xp_dir = resolve_xp_dir(cfg)
    files, _ = build_file_lists(cfg, xp_dir, dataset_folder)

    inference_set = getattr(cfg.inference, "set", "test")
    batch_size = cfg.inference.batch_size
    num_workers = getattr(cfg.inference, "num_workers", 4)
    loader = build_dataloader(
        files,
        mode=inference_set if inference_set in ("train", "val") else "test",
        batch_size=batch_size,
        num_workers=num_workers,
        augmentations=Augmentations(False, False, False),
        shuffle=False,
        drop_last=False,
    )

    checkpoints = resolve_checkpoints(cfg.inference.checkpoints, xp_dir)
    tta = build_tta_transforms(getattr(cfg.inference, "tta", ["none"]))
    eps = getattr(cfg.inference, "eps", 1e-8)
    snow_floor = getattr(cfg.inference, "snow_floor", None)
    compute_kl = getattr(cfg.inference, "compute_kl", inference_set != "test")

    ensemble_dists = []
    sample_ids = None
    gt_reference = None
    log_lines: List[str] = []

    for ckpt_path in checkpoints:
        print(f"Running model from {ckpt_path}")
        model = build_model(cfg, device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

        dists, ids, gt, tta_kls = predict_distributions_with_tta(
            model, loader, device=device, tta=tta, eps=eps, snow_floor=snow_floor
        )
        if sample_ids is None:
            sample_ids = ids
        elif sample_ids != ids:
            raise RuntimeError("Sample IDs differ between checkpoints; unexpected dataloader ordering change.")
        if gt is not None and gt_reference is None:
            gt_reference = gt
        ensemble_dists.append(dists)

        if compute_kl and gt is not None:
            model_kl = kl_divergence(gt, dists, eps=eps).mean().item()
            log_lines.append(f"model={ckpt_path.name}\tKL={model_kl:.6f}")
            for tta_name, tta_kl in tta_kls:
                log_lines.append(f"model={ckpt_path.name}\ttta={tta_name}\tKL={tta_kl:.6f}")

    assert len(ensemble_dists) > 0, "No predictions were collected."
    mean_dist = torch.stack(ensemble_dists, dim=0).mean(dim=0)

    # Expand back to 10-class format with leading zeros for no_data/clouds.
    full_dists = torch.zeros((mean_dist.shape[0], LandCoverData.N_CLASSES), dtype=mean_dist.dtype)
    full_dists[:, 2:] = mean_dist
    columns = LandCoverData.CLASSES
    df = pd.DataFrame(full_dists.numpy(), index=pd.Index(sample_ids, name="sample_id"), columns=columns)
    df.insert(0, "sample_id", df.index)
    sanity_checks(df.reset_index(drop=True), columns)

    # Optional validation KL (when masks are available).
    compute_kl = getattr(cfg.inference, "compute_kl", inference_set != "test")
    kl_value: Optional[float] = None
    if compute_kl and gt_reference is not None:
        kl_value = kl_divergence(gt_reference, mean_dist, eps=eps).mean().item()
        print(f"Validation KL (set={inference_set}): {kl_value:.6f}")
        log_lines.append(f"ensemble\tKL={kl_value:.6f}")

    out_csv = getattr(cfg.inference, "output_csv", None)
    if out_csv is None:
        out_csv = xp_dir / f"ensemble_{cfg.inference.set}_predicted.csv"
    out_csv = Path(out_csv).expanduser().resolve()
    print(f"Saving ensemble prediction CSV to {out_csv}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if kl_value is not None:
        kl_path = out_csv.parent / "val_loss.txt"
        kl_content = "\n".join(log_lines + [f"ensemble_mean\tKL={kl_value:.6f}"])
        kl_path.write_text(kl_content + "\n")
        print(f"Saved validation KL breakdown to {kl_path}")


if __name__ == "__main__":
    main()
