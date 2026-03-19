import torch
import torch.nn as nn
import torchvision.models as models

class GuavaModel(nn.Module):
    def __init__(self, handcrafted_dim=100):
        super().__init__()

        backbone = models.efficientnet_b0(weights="DEFAULT")
        self.cnn = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten()
        )

        self.cnn_proj = nn.Linear(1280, 256)
        self.hcf_proj = nn.Linear(handcrafted_dim, 64)

        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, features):
        cnn_out = self.cnn(image)
        cnn_out = self.cnn_proj(cnn_out)

        feat_out = self.hcf_proj(features)

        x = torch.cat([cnn_out, feat_out], dim=1)
        return self.fc(x).squeeze()