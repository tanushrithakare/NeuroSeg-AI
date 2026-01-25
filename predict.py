import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from model.model import NeuroSeg


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = NeuroSeg().to(device)
model.load_state_dict(torch.load("neuroseg.pth", map_location=device))
model.eval()

# Load image
img_path = "data/images/image_001.png"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))
image_norm = image / 255.0

# Convert to tensor
input_tensor = torch.tensor(image_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
input_tensor = input_tensor.to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.sigmoid(output).cpu().numpy()[0, 0]

# Threshold
mask = (pred > 0.5).astype(np.uint8)

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("MRI Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(image, cmap="gray")
plt.imshow(mask, alpha=0.5, cmap="jet")
plt.axis("off")

plt.show()
