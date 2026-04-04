"""
dataset.py
----------
GuavaDataset: loads images + handcrafted features + normalized labels.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from utils.feature_extraction import extract_features

LABEL_COLS = ['L', 'A', 'B', 'FIRMNESS', 'TA', 'TSS', 'PH', 'REMAINING DAYS TO RIPE']


class GuavaDataset(Dataset):
    def __init__(self, excel_path, image_dir, transform=None,
                 label_mean=None, label_std=None):
        self.df        = (pd.read_excel(excel_path)
                            .dropna(subset=['IMAGE '] + LABEL_COLS)
                            .reset_index(drop=True))
        self.image_dir = image_dir
        self.transform = transform
        self.label_mean = label_mean  # torch.Tensor (8,) or None
        self.label_std  = label_std   # torch.Tensor (8,) or None

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, img_name):
        img_name = str(img_name).strip()
        for candidate in [img_name,
                          img_name.replace('.jpeg', '.jpg'),
                          img_name.replace('.jpg', '.jpeg')]:
            p = os.path.join(self.image_dir, candidate)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Image not found: {img_name}")

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self._resolve_path(row['IMAGE '])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        features = torch.tensor(extract_features(img_path), dtype=torch.float32)

        labels = torch.tensor(row[LABEL_COLS].values.astype(float),
                              dtype=torch.float32)

        # Z-score normalize labels if stats provided
        if self.label_mean is not None and self.label_std is not None:
            labels = (labels - self.label_mean) / self.label_std

        return image, features, labels
