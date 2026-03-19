import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from utils.feature_extraction import extract_features

class GuavaDataset(Dataset):
    def __init__(self, excel_path, image_dir, transform=None):
        self.df = pd.read_excel(excel_path).dropna(subset=['IMAGE ', 'REMAINING DAYS TO RIPE']).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = str(row['IMAGE ']).strip().replace('.jpeg', '.jpg')
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        features = extract_features(img_path)
        features = torch.tensor(features, dtype=torch.float32)

        label = torch.tensor(row['REMAINING DAYS TO RIPE'], dtype=torch.float32)

        return image, features, label