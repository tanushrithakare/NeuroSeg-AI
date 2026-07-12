import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import BrainMRIDataset
from model.model import NeuroSeg
from utils import dice_loss_from_logits, dice_score_from_logits


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks) + dice_loss_from_logits(logits, masks)
            total_loss += loss.item()
            total_dice += dice_score_from_logits(logits, masks)

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_dataset = BrainMRIDataset(args.image_dir, args.mask_dir, image_size=args.image_size, augment=False)
    pairs = full_dataset.pairs
    
    random.seed(42)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    val_size = int(len(shuffled_pairs) * args.val_ratio)
    train_size = len(shuffled_pairs) - val_size
    
    train_pairs = shuffled_pairs[:train_size]
    val_pairs = shuffled_pairs[train_size:]
    
    train_dataset = BrainMRIDataset(args.image_dir, args.mask_dir, image_size=args.image_size, augment=True)
    train_dataset.pairs = train_pairs
    
    val_dataset = BrainMRIDataset(args.image_dir, args.mask_dir, image_size=args.image_size, augment=False)
    val_dataset.pairs = val_pairs

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = NeuroSeg(encoder_pretrained=args.encoder_pretrained).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss_from_logits(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_dice = validate(model, val_loader, bce, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output_path)
            print(f"Saved new best checkpoint to {output_path} (dice={best_dice:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroSeg on real MRI segmentation data.")
    parser.add_argument("--image-dir", required=True, help="Directory with MRI images")
    parser.add_argument("--mask-dir", required=True, help="Directory with segmentation masks")
    parser.add_argument("--output", default="checkpoints/neuroseg_best.pth", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder-pretrained", action="store_true", help="Use ImageNet-pretrained ResNet50 encoder")

    args = parser.parse_args()
    train(args)
