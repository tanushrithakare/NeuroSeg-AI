import os
import cv2
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(
            self.mask_dir,
            image_name.replace(".png", "_mask.png")
        )

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Missing image or mask for {image_name}")

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        image = image / 255.0
        mask = mask / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask
