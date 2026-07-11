# 🧠 NeuroSeg-AI

> **Attention-based deep learning system for semantic segmentation of brain tumors from MRI scans.**

NeuroSeg-AI is a PyTorch deep learning project that combines a **ResNet-50 encoder**, **ASPP (Atrous Spatial Pyramid Pooling)**, and a **CBAM (Convolutional Block Attention Module)** to accurately localize and segment tumor regions in grayscale MRI images. An interactive **Streamlit web dashboard** allows real-time inference with threshold control and overlay visualization.

---

## ✨ Features

- 🏗️ **Custom Segmentation Architecture** — ResNet-50 encoder + ASPP multi-scale context aggregation + CBAM spatial & channel attention + decoder
- 🎯 **Combined Loss Training** — `BCEWithLogitsLoss + Soft Dice Loss` for robust mask prediction
- 📊 **Kaggle LGG Dataset Support** — Ready-to-use data preparation script for the [LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- 🌐 **Streamlit Dashboard** — Upload MRI scans, view predicted masks, overlays, and an AI diagnostic summary
- 🔧 **Flexible CLI** — Full control over epochs, batch size, learning rate, validation split, and image size
- 💾 **Best Checkpoint Saving** — Automatically saves the model with the highest validation Dice score

---

## 🏛️ Model Architecture

```
MRI Image (Grayscale 1×H×W)
        │
        ▼ (replicated to 3 channels)
 ┌──────────────┐
 │  ResNet-50   │  ← Pre-trained ImageNet encoder (optional)
 │   Encoder    │
 └──────┬───────┘
        │  (multi-scale skip features x0–x4)
        ▼
 ┌──────────────┐
 │     ASPP     │  ← Atrous rates: 1×, 6×, 12×, 18×
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │     CBAM     │  ← Channel Attention + Spatial Attention
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │   Decoder    │  ← Fuses high-level features with low-level skip
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │  Classifier  │  ← 1×1 Conv → Binary segmentation mask
 └──────────────┘
```

| Component | Details |
|---|---|
| Encoder | ResNet-50 (torchvision, optional ImageNet weights) |
| Context Module | ASPP with dilation rates 6, 12, 18 |
| Attention | CBAM (Channel + Spatial Attention) |
| Loss | BCEWithLogitsLoss + Soft Dice Loss |
| Optimizer | AdamW (weight decay 1e-4) |
| Input size | 256 × 256 (grayscale, mapped to 3-channel) |
| Output | Single-channel binary mask |

---

## 📁 Project Structure

```
NeuroSeg-AI/
│
├── model/
│   ├── model.py          # NeuroSeg: full model assembly
│   ├── encoder.py        # ResNet-50 feature extractor
│   ├── aspp.py           # Atrous Spatial Pyramid Pooling
│   ├── attention.py      # CBAM (Channel + Spatial Attention)
│   └── decoder.py        # Skip-connection based decoder
│
├── app.py                # Streamlit web dashboard
├── train.py              # Training script (CLI)
├── predict.py            # CLI inference + matplotlib visualization
├── dataset.py            # BrainMRIDataset (PyTorch Dataset)
├── utils.py              # Dice score & Dice loss utilities
├── prepare_kaggle_data.py # Reorganize Kaggle LGG dataset
├── generate_dummy_data.py # Generate synthetic data for quick tests
│
├── data/                 # (created locally — not tracked by git)
│   ├── images/           # MRI images
│   └── masks/            # Corresponding binary masks
│
└── checkpoints/          # (created locally — not tracked by git)
    └── neuroseg_best.pth # Best model checkpoint
```

---

## 🗄️ Dataset

This project is designed to work with the **LGG MRI Segmentation** dataset from Kaggle.

> 📦 **Dataset:** [Brain MRI Segmentation — Mateusz Buda (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

The dataset contains **3,929 brain MRI images** and corresponding manually annotated masks for lower-grade glioma (LGG) patients from The Cancer Imaging Archive (TCIA).

### Downloading & Preparing the Dataset

**Step 1 — Download from Kaggle:**

Using the Kaggle CLI (recommended):
```bash
pip install kaggle
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d kaggle_data/
```

Or download manually from the Kaggle link above and extract the zip.

**Step 2 — Reorganize into NeuroSeg-AI format:**
```bash
python prepare_kaggle_data.py --source kaggle_data/lgg-mri-segmentation
```

This script scans all patient subdirectories and copies:
- `*_mask.tif` → `data/masks/`
- All other `.tif` files → `data/images/`

After running, your directory will look like:
```
data/
├── images/   ← ~3,900 MRI slices (.tif)
└── masks/    ← ~3,900 binary masks (.tif)
```

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+, pip

```bash
# Clone the repository
git clone https://github.com/tanushrithakare/NeuroSeg-AI.git
cd NeuroSeg-AI

# Install dependencies
pip install torch torchvision opencv-python matplotlib streamlit
```

> 💡 For GPU acceleration, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🚀 Usage

### 1. Prepare Data

Use the Kaggle dataset (recommended) or generate dummy data for a quick sanity check:

```bash
# Option A: Use Kaggle LGG dataset (see Dataset section above)
python prepare_kaggle_data.py --source /path/to/lgg-mri-segmentation

# Option B: Generate 5 synthetic images for quick testing
python generate_dummy_data.py
```

### 2. Train the Model

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

**Key training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--image-dir` | required | Path to MRI images directory |
| `--mask-dir` | required | Path to masks directory |
| `--output` | `checkpoints/neuroseg_best.pth` | Checkpoint save path |
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--val-ratio` | `0.2` | Fraction of data used for validation |
| `--image-size` | `256` | Input image size (square) |
| `--encoder-pretrained` | `False` | Use ImageNet pre-trained ResNet-50 |

Training prints per-epoch stats: `train_loss`, `val_loss`, `val_dice`. The best checkpoint is saved automatically.

### 3. CLI Inference

```bash
python predict.py
```

> Update the `img_path` variable in `predict.py` to point to your test image. Outputs a matplotlib figure showing: Original MRI | Predicted Mask | Overlay.

### 4. Launch the Web Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Features:
- Upload any MRI image (PNG, JPG, TIFF, BMP)
- Adjust **segmentation threshold** and **overlay transparency** via sidebar sliders
- View: Original MRI | Predicted Mask | Color Overlay
- AI Diagnostic Summary with estimated tumor coverage percentage
- Interactive Q&A chatbot section

---

## 🗂️ Dataset Format Reference

The model supports any custom dataset organized as follows:

```
data/
├── images/
│   ├── patient_001.tif
│   ├── patient_002.tif
│   └── ...
└── masks/
    ├── patient_001_mask.tif   ← preferred naming
    ├── patient_002_mask.tif
    └── ...
```

**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

**Mask naming convention** (auto-resolved for each image `stem`):
1. `{stem}_mask.png` / `{stem}_mask{ext}` ← preferred
2. `{stem}.png` / `{stem}{ext}`

---

## 📋 Requirements

| Package | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `torchvision` | ResNet-50 backbone |
| `opencv-python` | Image I/O and preprocessing |
| `matplotlib` | CLI visualization |
| `streamlit` | Web dashboard |
| `numpy` | Array operations |
| `Pillow` | Image handling in Streamlit |

---

## 🔬 Technical Notes

- **Grayscale Handling:** Single-channel MRI inputs are automatically replicated to 3 channels (`x.repeat(1, 3, 1, 1)`) before the ResNet-50 encoder.
- **Mask Binarization:** Masks are thresholded at pixel value `127` during training (`mask > 127`).
- **Validation:** A random 80/20 train-val split is used with a fixed seed (`42`) for reproducibility.
- **Best Model Tracking:** The checkpoint is updated only when validation Dice score improves, preventing overfitting checkpoints.
- **ASPP Rates:** Dilated convolutions at rates 6, 12, and 18 capture multi-scale context suitable for tumors of varying sizes.

---

## 📜 License

This project is released for academic and research purposes.

---

## 🙏 Acknowledgements

- Dataset: [LGG MRI Segmentation — Mateusz Buda et al.](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- Architecture inspired by DeepLabV3+ and CBAM attention mechanisms
- Built with [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/)
