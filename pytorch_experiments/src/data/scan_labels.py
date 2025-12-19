from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm


def main():
    repo_root = Path(__file__).resolve().parents[3]
    ds_root = repo_root / "pytorch_experiments" / "dataset"
    masks_dir = ds_root / "train" / "masks"

    mask_files = sorted(masks_dir.glob("*.tif"))
    print("n_masks:", len(mask_files))

    # scan rapide (tu peux augmenter à 2000 si besoin)
    N = min(17000, len(mask_files))

    labels = set()
    for p in tqdm(mask_files[:N], desc=f"Scanning {N} masks"):
        m = tifffile.imread(p)
        labels.update(np.unique(m).tolist())

    labels = sorted(labels)
    print("unique labels found:", labels)
    print("count:", len(labels))


if __name__ == "__main__":
    main()
