import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import tempfile, os
from models.model import GuavaModel
from utils.feature_extraction import extract_features

# Load model
model = GuavaModel()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(pil_image):
    if pil_image is None:
        return "Please upload an image."

    # Save to temp file so feature_extraction (cv2) can read it
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path)

    try:
        image_tensor = transform(pil_image).unsqueeze(0)

        features = extract_features(tmp_path)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(image_tensor, features_tensor).item()

        days = max(0, round(pred, 2))
        if days <= 1:
            status = "🔴 Overripe / Ready now"
        elif days <= 3:
            status = "🟡 Nearly ripe"
        else:
            status = "🟢 Still ripening"

        return f"Predicted days to ripe: {days}\nStatus: {status}"
    finally:
        os.unlink(tmp_path)


with gr.Blocks(title="Guava Ripeness Predictor") as demo:
    gr.Markdown("# 🍈 Guava Ripeness Predictor")
    gr.Markdown("Upload a guava image to predict how many days until it's ripe.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Guava Image")
        output_text = gr.Textbox(label="Prediction", lines=3)

    predict_btn = gr.Button("Predict", variant="primary")
    predict_btn.click(fn=predict, inputs=image_input, outputs=output_text)

    gr.Examples(
        examples=[["images/Picture1.jpg"]],
        inputs=image_input
    )

if __name__ == "__main__":
    demo.launch()
