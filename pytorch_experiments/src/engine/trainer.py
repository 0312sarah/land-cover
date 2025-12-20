from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from src.data.loader import build_dataloader, build_train_val_loaders
from src.data.transforms import build_transforms
from src.engine.early_stopping import EarlyStopping
from src.engine.losses import build_loss
from src.engine.metrics import confusion_to_metrics, save_confusion, update_confusion
from src.engine.schedulers import build_scheduler
from src.models.build import build_model


class Trainer:
    def __init__(self, cfg: dict, run_dir: Path, config_name: str | None = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.config_name = config_name

        # ----- device (CUDA if available, else CPU)
        cudnn.benchmark = True  # choose optimal kernels for fixed input sizes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # AMP only makes sense on CUDA
        self.use_amp = bool(cfg["train"].get("amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ----- data / transforms
        aug_cfg = cfg.get("augmentation", {})
        self.train_transform = build_transforms(aug_cfg, split="train")
        self.val_transform = build_transforms(aug_cfg, split="val")

        data_root = cfg["data"]["root"]
        batch_size = int(cfg["data"]["batch_size"])
        num_workers = int(cfg["data"].get("num_workers", 0))
        val_ratio = float(cfg["data"].get("val_ratio", 0.0))
        val_seed = int(cfg["data"].get("val_seed", cfg.get("seed", 42)))
        sampling_cfg = cfg["data"].get("sampling")

        if val_ratio > 0:
            self.train_loader, self.val_loader = build_train_val_loaders(
                root=data_root,
                batch_size=batch_size,
                num_workers=num_workers,
                val_ratio=val_ratio,
                seed=val_seed,
                train_transform=self.train_transform,
                val_transform=self.val_transform,
                sampling_cfg=sampling_cfg,
            )
        else:
            self.train_loader = build_dataloader(
                root=data_root,
                split="train",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                transform=self.train_transform,
            )
            self.val_loader = None

        # ----- model
        self.model = build_model(cfg["model"]).to(self.device)

        # ----- loss
        # Expect: logits (B,C,H,W) and mask (B,H,W) with values in [0..C-1]
        num_classes = cfg["model"].get("num_classes", 9)
        self.criterion = build_loss(cfg.get("loss"), num_classes=num_classes, device=self.device)

        # ----- optimizer
        lr = float(cfg["optim"]["lr"])
        wd = float(cfg["optim"].get("weight_decay", 0.0))
        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # ----- scheduler
        self.scheduler, self.scheduler_type = build_scheduler(self.optim, cfg.get("scheduler"))

        # ----- early stopping
        self.early_stopper = EarlyStopping.from_dict(cfg["train"].get("early_stop"))

        # ----- checkpointing config
        self.ckpt_cfg = cfg.get("checkpoint", {}) or {}
        self.save_best = bool(self.ckpt_cfg.get("save_best", True))
        self.save_last = bool(self.ckpt_cfg.get("save_last", True))
        self.save_every = int(self.ckpt_cfg.get("save_every", 0) or 0)
        self.monitor = "val_loss" if self.val_loader is not None else "train_loss"
        self.best_metric = float("inf") if self.monitor == "val_loss" else float("inf")
        self.best_epoch = 0

        # ----- output dirs
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        self.csv_path = self.run_dir / "logs.csv"
        self.metrics_path = self.run_dir / "metrics.json"
        self._init_csv()
        self._dump_config_txt()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_loss"])

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float | None):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, float(train_loss), float(val_loss) if val_loss is not None else ""])

    def _dump_config_txt(self):
        cfg_txt_path = self.run_dir / "config.txt"
        if cfg_txt_path.exists():
            return
        import yaml  # local import to avoid polluting module scope
        with open(cfg_txt_path, "w", encoding="utf-8") as f:
            if self.config_name:
                f.write(f"{self.config_name}\n")
            yaml.safe_dump(self.cfg, f, sort_keys=False)

    def _save_checkpoint(self, epoch: int, tag: str):
        ckpt_path = self.ckpt_dir / f"{tag}.pt"
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
        return ckpt_path

    def _save_metrics_summary(self, best_metric: float, best_epoch: int, last_epoch: int):
        summary = {
            "monitor": self.monitor,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "last_epoch": last_epoch,
            "scheduler": self.scheduler_type,
        }
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    @torch.no_grad()
    def _run_validation(self) -> tuple[float, dict]:
        """Full pass on val loader and restore training mode afterwards."""
        prev_mode = self.model.training
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        num_classes = self.cfg["model"].get("num_classes", 9)
        conf = torch.zeros((num_classes, num_classes), dtype=torch.int64).numpy()

        for images, masks in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            total_loss += float(loss.item())
            n_batches += 1

            preds = torch.argmax(logits, dim=1)
            conf = update_confusion(conf, preds, masks, num_classes=num_classes)

        metrics = confusion_to_metrics(conf)

        if prev_mode:
            self.model.train()

        return total_loss / max(n_batches, 1), metrics

    def fit(self):
        epochs = int(self.cfg["train"]["epochs"])
        log_every = int(self.cfg["train"].get("log_every", 50))
        val_each_log = bool(self.cfg["train"].get("val_each_log", True))

        global_step = 0
        last_val_loss = None
        last_epoch = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            steps_per_epoch = len(self.train_loader)
            last_train_loss = None
            train_loss_sum = 0.0
            train_loss_count = 0

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
                train_loss_sum += last_train_loss
                train_loss_count += 1

                if global_step % log_every == 0:
                    if self.val_loader is not None and val_each_log:
                        v_loss, v_metrics = self._run_validation()
                        last_val_loss = v_loss
                        self.writer.add_scalar("val/loss", v_loss, global_step)
                        self.writer.add_scalar("val/miou", v_metrics["mean"]["miou"], global_step)
                        self.writer.add_scalar("val/precision", v_metrics["mean"]["precision"], global_step)
                        self.writer.add_scalar("val/recall", v_metrics["mean"]["recall"], global_step)

                    val_loss_str = f"{last_val_loss:.4f}" if last_val_loss is not None else "N/A"
                    print(
                        f"epoch : ({epoch}/{epochs}) | step : ({step}/{steps_per_epoch}) | "
                        f"train loss : {last_train_loss:.4f} | val loss : {val_loss_str}"
                    )

                    self.writer.add_scalar("train/loss", last_train_loss, global_step)

            val_loss = None
            if self.val_loader is not None:
                val_loss, val_metrics = self._run_validation()
                last_val_loss = val_loss
                self.writer.add_scalar("val/loss", val_loss, global_step)
                self.writer.add_scalar("val/miou", val_metrics["mean"]["miou"], global_step)
                self.writer.add_scalar("val/precision", val_metrics["mean"]["precision"], global_step)
                self.writer.add_scalar("val/recall", val_metrics["mean"]["recall"], global_step)
                train_loss_str = f"{last_train_loss:.4f}" if last_train_loss is not None else "N/A"
                print(
                    f"epoch : ({epoch}/{epochs}) | step : ({steps_per_epoch}/{steps_per_epoch}) | "
                    f"train loss : {train_loss_str} | val loss : {val_loss:.4f}"
                )
            last_epoch = epoch

            # epoch-level logging
            avg_train_loss = train_loss_sum / max(train_loss_count, 1)
            self._log_epoch(epoch=epoch, train_loss=avg_train_loss, val_loss=val_loss)
            self.writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar("val/epoch_loss", val_loss, epoch)
                self.writer.add_scalar("val/epoch_miou", val_metrics["mean"]["miou"], epoch)
                self.writer.add_scalar("val/epoch_precision", val_metrics["mean"]["precision"], epoch)
                self.writer.add_scalar("val/epoch_recall", val_metrics["mean"]["recall"], epoch)

            # scheduler step
            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    metric_for_sched = val_loss if val_loss is not None else avg_train_loss
                    self.scheduler.step(metric_for_sched)
                else:
                    self.scheduler.step()

            # checkpointing (best/periodic/last)
            metric_for_ckpt = val_loss if val_loss is not None else avg_train_loss
            is_best = metric_for_ckpt < self.best_metric
            if is_best:
                self.best_metric = metric_for_ckpt
                self.best_epoch = epoch
                if self.save_best:
                    self._save_checkpoint(epoch, tag="best")
                    if val_loss is not None and "confusion" in val_metrics:
                        save_confusion(
                            np.array(val_metrics["confusion"], dtype=int),
                            self.run_dir / "val_confusion_best.json",
                        )

            if self.save_every > 0 and (epoch % self.save_every == 0):
                self._save_checkpoint(epoch, tag=f"epoch_{epoch:03d}")

            if self.save_last:
                self._save_checkpoint(epoch, tag="last")

            # early stopping
            if self.early_stopper.step(metric_for_ckpt, epoch):
                print(
                    f"[EARLY STOP] no improvement for {self.early_stopper.cfg.patience} epochs "
                    f"(best @ epoch {self.early_stopper.best_epoch})"
                )
                break

        self.writer.close()
        # persist metrics summary for analysis
        self._save_metrics_summary(best_metric=self.best_metric, best_epoch=self.best_epoch, last_epoch=last_epoch)
