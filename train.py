"""
train.py
--------
Two-phase training strategy for GuavaNet:
  Phase 1 (epochs 1–15) : Backbone frozen, train head + HCF branch only
  Phase 2 (epochs 16–40): Unfreeze last 3 EfficientNet blocks, fine-tune all

Features:
  - Z-score label normalization (saved to label_stats.pt for inference)
  - SmoothL1Loss for robustness to outliers
  - AdamW + ReduceLROnPlateau
  - Per-target MAE tracking
  - Best model saved by validation loss
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from models.model import GuavaNet
from utils.dataset import GuavaDataset

# ── Config ───────────────────────────────────────────────────────────────────
EXCEL_PATH   = "images/label.xlsx"
IMAGE_DIR    = "images"
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 25
BATCH_SIZE   = 16
LR_PHASE1    = 1e-3
LR_PHASE2    = 3e-4
VAL_SPLIT    = 0.20
SEED         = 42

LABEL_COLS = ['L', 'A', 'B', 'FIRMNESS', 'TA', 'TSS', 'PH', 'REMAINING DAYS TO RIPE']

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset & Label Normalization ─────────────────────────────────────────────
full_df = pd.read_excel(EXCEL_PATH).dropna(subset=['IMAGE '] + LABEL_COLS)
label_values = full_df[LABEL_COLS].values.astype(np.float32)

label_mean = torch.tensor(label_values.mean(axis=0), dtype=torch.float32)
label_std  = torch.tensor(label_values.std(axis=0)  + 1e-8, dtype=torch.float32)

# Save stats so app.py / predict.py can inverse-transform
torch.save({'mean': label_mean, 'std': label_std}, 'label_stats.pt')
print(f"Label means : {label_mean.numpy().round(4)}")
print(f"Label stds  : {label_std.numpy().round(4)}")

full_dataset = GuavaDataset(
    EXCEL_PATH, IMAGE_DIR,
    transform=train_transform,
    label_mean=label_mean,
    label_std=label_std
)

val_size   = max(1, int(VAL_SPLIT * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

# Val set uses val_transform — patch the transform
val_set.dataset = GuavaDataset(
    EXCEL_PATH, IMAGE_DIR,
    transform=val_transform,
    label_mean=label_mean,
    label_std=label_std
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

print(f"Train: {train_size} | Val: {val_size}")

# ── Model ─────────────────────────────────────────────────────────────────────
model    = GuavaNet(handcrafted_dim=190, num_outputs=8, dropout=0.4)
loss_fn  = nn.SmoothL1Loss()
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True, optimizer=None):
    model.train() if train else model.eval()
    total_loss = 0
    per_target_mae = torch.zeros(len(LABEL_COLS))

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img, feat, labels in loader:
            img, feat, labels = img.to(device), feat.to(device), labels.to(device)
            preds = model(img, feat)
            loss  = loss_fn(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            # MAE in normalized space
            per_target_mae += (preds - labels).abs().mean(dim=0).cpu().detach()

    n = len(loader)
    return total_loss / n, (per_target_mae / n).numpy()


def print_per_target(mae_arr, prefix="Val MAE"):
    parts = [f"{name}: {val:.4f}" for name, val in zip(LABEL_COLS, mae_arr)]
    print(f"  {prefix} | " + " | ".join(parts))


# ── Phase 1: Frozen Backbone ──────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 1 — Training head with frozen backbone")
print("="*60)

model.freeze_backbone()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE1, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=4
)

best_val_loss = float('inf')

for epoch in range(1, PHASE1_EPOCHS + 1):
    tr_loss, _       = run_epoch(train_loader, train=True,  optimizer=optimizer)
    val_loss, val_mae = run_epoch(val_loader,   train=False)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
          f"Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
    print_per_target(val_mae)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        print(f"  ✓ Saved (val={best_val_loss:.4f})")

# ── Phase 2: Fine-tune last 3 blocks ─────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 — Fine-tuning last 3 EfficientNet blocks")
print("="*60)

model.unfreeze_backbone(last_n_blocks=3)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE2, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

for epoch in range(1, PHASE2_EPOCHS + 1):
    tr_loss, _       = run_epoch(train_loader, train=True,  optimizer=optimizer)
    val_loss, val_mae = run_epoch(val_loader,   train=False)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
          f"Train: {tr_loss:.4f} | Val: {val_loss:.4f}")
    print_per_target(val_mae)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        print(f"  ✓ Saved (val={best_val_loss:.4f})")

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
print("Model saved → model.pth | Label stats → label_stats.pt")
