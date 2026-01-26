# ENS Challenge Data 2021: Land cover predictive modeling from satellite images

This repository stores the code for the benchmark model of the Challenge Data competition "Land cover predictive modeling from satellite images" (Preligens).
The proposed benchmark model is a deep neural network trained on the proxy task of semantic segmentation of the land cover labels at the pixel level. The network has a U-Net architecture ([Ronneberger et al 2015](https://arxiv.org/abs/1505.04597)).

## Repository structure

- TensorFlow baseline: `tensorflow_baseline/`
- PyTorch baseline: `pytorch_baseline/` (recommended for most experiments)

See `pytorch_baseline/README.md` for up-to-date training/inference details and the additional model variants.

## Data folder

You can download the data as an archive containing the training images and masks, as well as the test images, from the challenge page.

The dataset folder should be like this:
```
dataset_UNZIPPED
|-- test
|   |-- images
|   |   |-- 10087.tif
|   |   |-- 10088.tif
|   |   |-- 10089.tif
|   |   |-- 10090.tif
        ... (5043 files)
|-- train
    |-- images
    |   |-- 10000.tif
    |   |-- 10001.tif
    |   |-- 10002.tif
    |   |-- 10003.tif
        ... (18491 files)
    |-- masks
        |-- 10000.tif
        |-- 10001.tif
        |-- 10002.tif
        |-- 10003.tif
        ... (18491 files)
```

The images are 16-bit GeoTIFF files of size (256, 256, 4) and the masks are 8-bit GeoTIFF files of size (256, 256).
Every sample has an identifier used in the CSVs in a column named `sample_id`.

## Python environment

The file `environment.yml` is an exported conda environment with all the dependencies for this project.
You can recreate it with:
```
conda env create -f environment.yml
```

For Windows users:
```
conda env create -f environment_windows.yml
conda activate landcover-win
```

## Code usage (TensorFlow baseline)

The `framework` package contains scripts to train the model, then use the trained model to perform predictions over the testing set, and finally evaluate those predictions against corresponding ground truth labels.

### Train
```
ipython framework/train.py -- --config config.yaml
```
See `train_config.yaml` for parameters and a working example.

### Predict
```
ipython framework/infer.py -- --config config.yaml
```

### Evaluate
```
ipython framework/eval.py -- --gt-file path/to/labels.csv --pred-file path/to/predicted.csv -o /path/to/save.csv
```

## Model information (TensorFlow baseline)

U-Net is composed of a contractive path and expansive path. The contractive path diminishes the spatial resolution by repeated (3,3) convolutions and (2,2) max pooling. Every step in the expansive path consists of (2,2) transposed convolutions that expand the spatial resolution, concatenation with the corresponding feature map, followed by (3,3) convolutions. The last layer uses a (1,1) convolution to produce class logits.

The benchmark model is a small network made of ~984k trainable parameters. The total number of convolution layers is 21, made of 64 feature maps in the contractive path, and 64 or 96 feature maps in the expansive path.

The loss function used is cross-entropy, with weights assigned to every class (inverse frequency). The special classes "no_data" and "clouds" are set to weight 0 to avoid learning to predict them.

As data augmentations during training, we used basic flips and rotations (90°, 270°). The model was trained for 90 epochs on a Nvidia Tesla P100-PCIE-16GB.
