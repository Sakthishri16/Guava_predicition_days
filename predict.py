import torch
import torchvision.transforms as transforms
from PIL import Image
from models.model import GuavaModel
from utils.feature_extraction import extract_features

# Load model
model = GuavaModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    features = extract_features(image_path)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(image, features)

    return pred.item()


# Test
print("Predicted days:", predict("images/Picture1.jpg"))