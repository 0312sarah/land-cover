# TensorFlow Baseline – ENS Challenge Data 2021 (Land Cover)

This directory contains the **reference (baseline)** implementation for the Challenge Data competition
["Land cover predictive modeling from satellite images"](https://challengedata.ens.fr/challenges/48) provided by Preligens.

The baseline model is a deep neural network trained on the “proxy” task of semantic segmentation of land cover labels at the pixel level.
The network has a U-Net architecture (Ronneberger et al., 2015).

---

## 1) Dataset

Download the dataset archive from the challenge page and unzip it locally.

### Recommended location (repository root)

Place the dataset in the repository root under `data/` so it can be shared across implementations (TensorFlow baseline and future PyTorch experiments):

land-cover/
├── data/
│ └── dataset/
│ ├── test/
│ │ └── images/
│ │ ├── 10087.tif
│ │ └── ...
│ └── train/
│ ├── images/
│ │ ├── 10000.tif
│ │ └── ...
│ └── masks/
│ ├── 10000.tif
│ └── ...
└── tensorflow_baseline/