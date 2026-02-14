import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError("Image directory or mask directory does not exist.")

        valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        self.pairs = []

        for image_path in sorted(self.image_dir.iterdir()):
            if image_path.suffix.lower() not in valid_ext:
                continue

            stem = image_path.stem
            mask_candidates = [
                self.mask_dir / f"{stem}_mask{image_path.suffix}",
                self.mask_dir / f"{stem}{image_path.suffix}",
                self.mask_dir / f"{stem}_mask.png",
                self.mask_dir / f"{stem}.png",
            ]
            mask_path = next((p for p in mask_candidates if p.exists()), None)

            if mask_path is not None:
                self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise RuntimeError("No image/mask pairs found. Check naming and paths.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Could not read image/mask: {image_path.name}")

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = torch.tensor(image / 255.0, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor((mask > 127).astype("float32"), dtype=torch.float32).unsqueeze(0)

        return image, mask
