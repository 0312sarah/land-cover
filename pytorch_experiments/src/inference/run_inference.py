import argparse
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
import tifffile

from src.data.dataset import LandCoverDataset
from src.data.transforms import build_transforms
from src.models.build import build_model


def _select_device(device_str: str | None) -> torch.device:
    if device_str is None or device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg["model"]).to(device)
    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state") if isinstance(state, dict) else None
    if state_dict is None:
        state_dict = state if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def run_inference(cfg: dict):
    device = _select_device(cfg.get("device"))

    # data / transforms
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    split = data_cfg.get("split", "test")
    batch_size = int(data_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))

    transform = build_transforms(cfg.get("augmentation"), split="val")

    dataset = LandCoverDataset(root=root, split=split, transform=transform)
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    if num_workers > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **dl_kwargs)

    # model
    model = _load_model(cfg, device=device)

    # output
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    map_to_original = bool(cfg["output"].get("map_to_original_labels", True))

    idx_base = 0
    for batch_idx, images in enumerate(loader):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        if map_to_original:
            preds = preds + 1  # back to original labels 1..num_classes
        preds_np = preds.cpu().numpy().astype("uint8")

        batch_files = [dataset.images_files[idx_base + i] for i in range(preds_np.shape[0])]
        for arr, src_path in zip(preds_np, batch_files):
            out_path = out_dir / Path(src_path).name
            tifffile.imwrite(out_path, arr)
        idx_base += preds_np.shape[0]

        print(f"[{idx_base}/{len(dataset)}] saved batch {batch_idx + 1}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on the test split.")
    parser.add_argument("--config", required=True, help="Path to inference YAML config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_inference(cfg)


if __name__ == "__main__":
    main()
