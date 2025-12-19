from __future__ import annotations

from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from src.data.loader import build_dataloader, build_train_val_loaders
from src.models.build import build_model


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)

        # ----- device (CUDA if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # AMP only makes sense on CUDA
        self.use_amp = bool(cfg["train"].get("amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ----- data
        data_root = cfg["data"]["root"]
        batch_size = int(cfg["data"]["batch_size"])
        num_workers = int(cfg["data"].get("num_workers", 0))
        val_ratio = float(cfg["data"].get("val_ratio", 0.0))
        val_seed = int(cfg["data"].get("val_seed", cfg.get("seed", 42)))

        if val_ratio > 0:
            self.train_loader, self.val_loader = build_train_val_loaders(
                root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                val_ratio=val_ratio,
                seed=val_seed,
            )
        else:
            self.train_loader = build_dataloader(
                root=data_root,
                split="train",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )
            self.val_loader = None

        # ----- model
        self.model = build_model(cfg["model"]).to(self.device)

        # ----- loss
        # Expect: logits (B,C,H,W) and mask (B,H,W) with values in [0..C-1]
        self.criterion = nn.CrossEntropyLoss()

        # ----- optimizer
        lr = float(cfg["optim"]["lr"])
        wd = float(cfg["optim"].get("weight_decay", 0.0))
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # ----- output dirs
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        self.csv_path = self.run_dir / "logs.csv"
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "global_step", "split", "loss"])

    def _log_csv(self, epoch: int, global_step: int, split: str, loss: float):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, global_step, split, float(loss)])

    @torch.no_grad()
    def _validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(self):
        epochs = int(self.cfg["train"]["epochs"])
        log_every = int(self.cfg["train"].get("log_every", 50))

        global_step = 0
        last_val_loss = None

        for epoch in range(1, epochs + 1):
            self.model.train()
            steps_per_epoch = len(self.train_loader)
            last_train_loss = None

            for step, (images, masks) in enumerate(self.train_loader, start=1):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                self.optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()

                global_step += 1
                last_train_loss = float(loss.item())

                if global_step % log_every == 0:
                    val_loss_str = f"{last_val_loss:.4f}" if last_val_loss is not None else "N/A"
                    print(
                        f"epoch : ({epoch}/{epochs}) | step : ({step}/{steps_per_epoch}) | "
                        f"train loss : {last_train_loss:.4f} | val loss : {val_loss_str}"
                    )

                    self._log_csv(epoch=epoch, global_step=global_step, split="train", loss=last_train_loss)
                    self.writer.add_scalar("train/loss", last_train_loss, global_step)

            val_loss = None
            if self.val_loader is not None:
                val_loss = self._validate_epoch()
                last_val_loss = val_loss
                self.writer.add_scalar("val/loss", val_loss, global_step)
                self._log_csv(epoch=epoch, global_step=global_step, split="val", loss=val_loss)
                train_loss_str = f"{last_train_loss:.4f}" if last_train_loss is not None else "N/A"
                print(
                    f"epoch : ({epoch}/{epochs}) | step : ({steps_per_epoch}/{steps_per_epoch}) | "
                    f"train loss : {train_loss_str} | val loss : {val_loss:.4f}"
                )

            # save checkpoint every epoch
            ckpt_path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optim.state_dict(),
                    "cfg": self.cfg,
                },
                ckpt_path,
            )
            print(f"[OK] saved checkpoint: {ckpt_path}")

        self.writer.close()
