# PyTorch Baseline (Preligens Land Cover Challenge)

Clean PyTorch reimplementation of the TensorFlow baseline in `tensorflow_baseline/`. The code matches the same U-Net architecture, preprocessing, train/val split logic, optimizer defaults, and provides YAML-driven training and inference/submission generation.

## Commands
- Train: `python -m baseline.train --config configs/train/unet_ce_global_kl.yaml`
- Inference / submission: `python -m baseline.infer --config configs/infer/infer.yaml`

Run from inside the `pytorch_baseline/` folder (so the `baseline` package is on `PYTHONPATH`). Ensure your dataset is available and pointed to by `data.root` in the YAML (no data copying performed).

## What matches the TensorFlow baseline
- U-Net: constant 64 conv filters and 96 up-conv filters, `num_layers=2`, BN→Conv(3x3, relu, same), BN→ConvTranspose(3x3, stride2, output_padding=1, relu), final 1x1 conv to 10 classes.
- Preprocessing: TIFF reading; normalization by dividing by `TRAIN_PIXELS_MAX=24356`; augmentations with 0.5 prob H/V flips, then 0.5 rot90 or else-if 0.5 rot270.
- Train/val split: shuffle with seed; optional fixed `val_samples.csv`; otherwise hold-out first 10% after shuffle (keeps original bookkeeping bug: `trainset_size = len(train)-val_size`).
- Optimizer/scheduler: Adam (lr from YAML), ReduceLROnPlateau (`patience=20`, `factor=0.5`).
- Logging: `log.txt` per run with lines `epoch,train_loss,val_loss,val_kl`.
- Validation metric `val_kl`: per-image distributions (drop classes 0/1, renormalize, eps=1e-8) and mean KL.
- Best model metrics: the best checkpoint by lowest `val_kl` is saved to `checkpoints/best.pt`, and per-class precision/recall on the validation set are written to `metrics/metrics.txt`.

## Submission CSV
- Uses soft probabilities: softmax logits → mean over H,W → drop classes 0/1 → add eps=1e-8 → renormalize (8 classes), then pads leading zeros for `no_data` and `clouds` to match the TF baseline column order.
- Columns: `sample_id,no_data,clouds,artificial,cultivated,broadleaf,coniferous,herbaceous,natural,snow,water`.
- Sanity checks ensure correct columns, non-negative finite values, row sums within 1e-5 of 1, and print a preview/min/max.

## Config highlights
- YAML driven for training and inference (model/loss/augmentation selection).
- Loss options: `cross_entropy`, `ce_global_kl`, `ce_dice_global_kl` with optional `class_weights`, `ignore_index`, `kl_weight`, `dice_weight`, `exclude_classes`, `smooth`.
- Augmentations toggled via YAML (`hflip`, `vflip`, `rotate`).
- Data root configurable (`data.root`); no dataset duplication. If you need a local shortcut, create a symlink yourself pointing to the dataset folder.

## Outputs
- Run directory under `training.xp_rootdir` (timestamped unless `xp_name` set) containing checkpoints (`checkpoints/epoch{N}.pt`), `log.txt`, `config.yaml`, and `val_samples.csv`.
- Inference: set `inference.checkpoint_epoch` to a number or `inference.checkpoint: best` to load `checkpoints/best.pt`; outputs default to `epoch{checkpoint_epoch}_{set}_predicted.csv` (or `best_{set}_predicted.csv` if using best) inside the run directory unless `inference.output_csv` is provided.
