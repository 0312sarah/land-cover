from pathlib import Path

import numpy as np 
import torch
from torch.utils.data import Dataset
import tifffile 

class LandCoverDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.label_map = {lab: lab - 1 for lab in range(1, 10)}
        self.num_classes = 9

        assert split in {"train", "test"}

        self.images_dir = self.root / split / "images"
        self.mask_dir = None
        if split == "train":
            self.mask_dir = self.root / split / "masks"

        self.images_files = sorted(self.images_dir.glob("*.tif"))
        if len(self.images_files) == 0:
            raise FileNotFoundError(f"No .tif found in {self.images_dir}")


    def __len__(self) -> int : 
        return len(self.images_files)
    
    def __getitem__(self, idx:int):
        # load l'image
        img_path = self.images_files[idx]
        image = tifffile.imread(img_path) # (H,W,C)

        image = image.astype(np.float32)
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image)

        if self.split == "train":
            mask_path = self.mask_dir / img_path.name
            mask = tifffile.imread(mask_path) # (H,W)

            mapped = np.full(mask.shape, fill_value=-1, dtype=np.int64)
            for src, dst in self.label_map.items():
                mapped[mask == src] = dst

            # sécurité : aucun pixel non mappé
            if (mapped < 0).any():
                raise ValueError("Mask contains labels not in label_map")

            mask = torch.from_numpy(mapped).long()

            return image,mask

        return image 

