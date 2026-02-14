# NeuroSeg-AI

Attention-based deep learning model for brain tumor semantic segmentation from MRI scans.

## Train on real data

### 1) Organize your dataset
Create two folders:

- `data/images/` — MRI slices (grayscale recommended)
- `data/masks/` — binary masks for each image

Supported file extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`.

Mask matching rules per image stem (example image: `patient_001.png`):

- `patient_001_mask.png` (preferred)
- `patient_001.png`

### 2) Install dependencies

```bash
pip install torch torchvision opencv-python matplotlib
```

### 3) Start training

```bash
python train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --output checkpoints/neuroseg_best.pth \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --encoder-pretrained
```

### 4) Run prediction

Update checkpoint path in `predict.py` if needed, then run:

```bash
python predict.py
```

## Notes

- The model accepts grayscale images and internally maps them to 3-channel input.
- Training uses `BCEWithLogitsLoss + Dice loss` and saves the best validation checkpoint by Dice score.
