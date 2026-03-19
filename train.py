import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.model import GuavaModel
from utils.dataset import GuavaDataset

# Paths
EXCEL_PATH = "images/label.xlsx"
IMAGE_DIR = "images"

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
dataset = GuavaDataset(EXCEL_PATH, IMAGE_DIR, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = GuavaModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Training
for epoch in range(5):
    for img, feat, label in loader:
        pred = model(img, feat)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "model.pth")