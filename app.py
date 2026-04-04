import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import tempfile, os
from models.model import GuavaNet
from utils.feature_extraction import extract_features

LABEL_NAMES = ['L', 'A', 'B', 'FIRMNESS', 'TA', 'TSS', 'PH', 'REMAINING DAYS TO RIPE']

# ── Load model + label stats ──────────────────────────────────────────────────
model = GuavaNet(handcrafted_dim=190, num_outputs=8)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

stats      = torch.load("label_stats.pt", map_location="cpu")
label_mean = stats['mean']
label_std  = stats['std']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(pil_image):
    if pil_image is None:
        return tuple(["—"] * len(LABEL_NAMES))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path)

    try:
        img_t  = transform(pil_image).unsqueeze(0)
        feat_t = torch.tensor(extract_features(tmp_path),
                              dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            preds = model(img_t, feat_t).squeeze()

        preds = preds * label_std + label_mean   # inverse z-score
        return tuple(f"{v:.4f}" for v in preds.tolist())
    finally:
        os.unlink(tmp_path)


css = """
.title-block { text-align:center; padding:20px 0 10px; }
.title-block h1 { font-size:2.2em; color:#2d6a2d; margin-bottom:4px; }
.title-block p  { color:#555; font-size:1em; }
.predict-btn { background:#2d6a2d !important; color:white !important;
               font-size:1.1em !important; border-radius:10px !important; }
"""

with gr.Blocks(title="Guava Quality Predictor", css=css,
               theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="title-block">
        <h1>🍈 Guava Quality Predictor</h1>
        <p>Upload a guava image to predict colour, chemical, physical and ripeness parameters.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 Upload Guava Image",
                                   height=320)
            predict_btn = gr.Button("🔍 Predict", variant="primary",
                                    elem_classes="predict-btn")

        with gr.Column(scale=2):
            gr.Markdown("### 🎨 Colour (LAB)")
            with gr.Row():
                out_L = gr.Textbox(label="💡 L  — Lightness",      interactive=False)
                out_A = gr.Textbox(label="🔴 A  — Green / Red",    interactive=False)
                out_B = gr.Textbox(label="🟡 B  — Blue / Yellow",  interactive=False)

            gr.Markdown("### 🧪 Chemical Properties")
            with gr.Row():
                out_TA  = gr.Textbox(label="🍋 TA  (%)",      interactive=False)
                out_TSS = gr.Textbox(label="🍬 TSS  (°Brix)", interactive=False)
                out_PH  = gr.Textbox(label="⚗️ pH",           interactive=False)

            gr.Markdown("### 📊 Physical & Ripeness")
            with gr.Row():
                out_FIRM = gr.Textbox(label="💪 Firmness  (N)",          interactive=False)
                out_DAYS = gr.Textbox(label="📅 Remaining Days to Ripe", interactive=False)

    outputs = [out_L, out_A, out_B, out_FIRM, out_TA, out_TSS, out_PH, out_DAYS]
    predict_btn.click(fn=predict, inputs=image_input, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
