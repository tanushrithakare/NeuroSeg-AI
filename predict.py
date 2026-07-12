import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.model import NeuroSeg


def main():
    parser = argparse.ArgumentParser(description="Run inference with NeuroSeg on a single MRI image.")
    parser.add_argument(
        "--image",
        type=str,
        default="data/images/TCGA_CS_4941_19960909_10.tif",
        help="Path to the input MRI image.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/neuroseg_best.pth",
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binarising the predicted mask (default: 0.5).",
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = NeuroSeg().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f"Could not load image at '{args.image}'. "
            "Please check the path and try again."
        )

    image_resized = cv2.resize(image, (256, 256))
    image_norm = image_resized / 255.0

    # ── Convert to tensor ─────────────────────────────────────────────────────
    input_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # ── Predict with Test-Time Augmentation (TTA) ─────────────────────────────
    with torch.no_grad():
        # 1. Original
        p1 = torch.sigmoid(model(input_tensor))

        # 2. Horizontal flip
        input_hf = torch.flip(input_tensor, dims=[3])
        p2 = torch.flip(torch.sigmoid(model(input_hf)), dims=[3])

        # 3. Vertical flip
        input_vf = torch.flip(input_tensor, dims=[2])
        p3 = torch.flip(torch.sigmoid(model(input_vf)), dims=[2])

        # Average consensus
        pred = ((p1 + p2 + p3) / 3.0).cpu().numpy()[0, 0]

    # ── Threshold ─────────────────────────────────────────────────────────────
    mask = (pred > args.threshold).astype(np.uint8)

    # ── Visualization ─────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("MRI Image")
    plt.imshow(image_resized, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(image_resized, cmap="gray")
    plt.imshow(mask, alpha=0.5, cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
