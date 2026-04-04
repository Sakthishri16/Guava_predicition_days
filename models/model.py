"""
model.py
--------
GuavaNet: EfficientNet-B0 backbone fused with handcrafted features.
Multi-output regression head predicts 8 quality parameters.

Architecture:
  CNN branch  : EfficientNet-B0 → 1280-d → BN → ReLU → Dropout → 256-d
  HCF branch  : 190-d → BN → ReLU → 64-d
  Fusion head : 320-d → BN → ReLU → Dropout → 128-d → ReLU → 8-d
"""

import torch
import torch.nn as nn
import torchvision.models as models


class GuavaNet(nn.Module):
    def __init__(self, handcrafted_dim: int = 190, num_outputs: int = 8,
                 dropout: float = 0.4):
        super().__init__()

        # ── CNN Backbone (EfficientNet-B0) ───────────────────────────────────
        backbone = models.efficientnet_b0(weights="DEFAULT")
        self.features = backbone.features   # output: (B, 1280, 7, 7)
        self.pool     = backbone.avgpool    # output: (B, 1280, 1, 1)

        # Freeze all backbone layers initially (Phase 1 training)
        for p in self.features.parameters():
            p.requires_grad = False

        # CNN projection branch
        self.cnn_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Handcrafted Feature Branch ───────────────────────────────────────
        self.hcf_branch = nn.Sequential(
            nn.Linear(handcrafted_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # ── Fusion Regression Head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
        )

        # Weight init for new layers
        self._init_weights()

    def _init_weights(self):
        for m in [self.cnn_branch, self.hcf_branch, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def freeze_backbone(self):
        """Phase 1: train only head + HCF branch."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 3):
        """Phase 2: unfreeze last N blocks of EfficientNet for fine-tuning."""
        # EfficientNet-B0 has 9 feature blocks (indices 0–8)
        blocks = list(self.features.children())
        for block in blocks[-(last_n_blocks):]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, image: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        # CNN path
        x = self.features(image)
        x = self.pool(x)
        x = self.cnn_branch(x)

        # HCF path
        h = self.hcf_branch(features)

        # Fuse and predict
        out = self.head(torch.cat([x, h], dim=1))
        return out   # (B, 8)


# Keep old name as alias for backward compatibility with app.py
GuavaModel = GuavaNet
