"""
predict.py
----------
Run inference on a single image or all images in the images/ folder.
Inverse-transforms z-score normalized predictions back to original scale.

Usage:
  python predict.py                        # predict all images
  python predict.py images/Picture1.jpg    # predict single image
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from models.model import GuavaNet
from utils.feature_extraction import extract_features
import os, glob, sys
import pandas as pd

LABEL_NAMES = ['L', 'A', 'B', 'FIRMNESS', 'TA', 'TSS', 'PH', 'REMAINING DAYS TO RIPE']

# ── Load model ────────────────────────────────────────────────────────────────
model = GuavaNet(handcrafted_dim=190, num_outputs=8)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ── Load label stats for inverse transform ────────────────────────────────────
stats = torch.load("label_stats.pt", map_location="cpu")
label_mean = stats['mean']
label_std  = stats['std']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)

    feat_t = torch.tensor(extract_features(image_path),
                          dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_t, feat_t).squeeze()

    # Inverse z-score
    preds = preds * label_std + label_mean
    return dict(zip(LABEL_NAMES, preds.tolist()))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = predict(sys.argv[1])
        print(f"\nPredictions for: {sys.argv[1]}")
        for k, v in result.items():
            print(f"  {k:30s}: {v:.4f}")
    else:
        image_files = sorted(
            glob.glob("images/*.jpg") +
            glob.glob("images/*.jpeg") +
            glob.glob("images/*.png")
        )
        print(f"Predicting {len(image_files)} images...\n")
        rows = []
        for img_path in image_files:
            result = predict(img_path)
            result['IMAGE'] = os.path.basename(img_path)
            rows.append(result)

        df = pd.DataFrame(rows, columns=['IMAGE'] + LABEL_NAMES)
        print(df.to_string(index=False))
        df.to_excel("predictions.xlsx", index=False)
        print("\nSaved → predictions.xlsx")
