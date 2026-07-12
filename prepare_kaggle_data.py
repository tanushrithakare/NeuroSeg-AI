import os
import shutil
from pathlib import Path

def prepare_lgg_dataset(kaggle_dir_path):
    """
    Reorganizes the LGG MRI Segmentation dataset from Kaggle into the 
    data/images and data/masks folders expected by NeuroSeg-AI.
    """
    source_dir = Path(kaggle_dir_path)
    if not source_dir.exists():
        print(f"Error: Source directory {kaggle_dir_path} does not exist.")
        return

    # Define output directories
    img_out_dir = Path("data/images")
    mask_out_dir = Path("data/masks")

    # Create output directories if they don't exist
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_masks = 0

    print("Scanning Kaggle dataset and copying files...")
    
    # Walk through the extracted Kaggle directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.endswith(".tif"):
                continue
                
            file_path = Path(root) / file
            
            # Check if it's a mask or an image
            if file.endswith("_mask.tif"):
                shutil.copy2(file_path, mask_out_dir / file)
                copied_masks += 1
            else:
                shutil.copy2(file_path, img_out_dir / file)
                copied_images += 1

    print("[DONE] Data preparation complete!")
    print(f"Copied {copied_images} images to {img_out_dir}")
    print(f"Copied {copied_masks} masks to {mask_out_dir}")
    print("You can now run: python train.py --image-dir data/images --mask-dir data/masks")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Kaggle LGG dataset")
    parser.add_argument("--source", type=str, required=True, help="Path to extracted Kaggle dataset folder")
    args = parser.parse_args()
    
    prepare_lgg_dataset(args.source)
