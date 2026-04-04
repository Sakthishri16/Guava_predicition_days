"""
app.py — Guava Quality Predictor (lightweight: onnxruntime only, no torch)
"""
import gradio as gr
import onnxruntime as ort
import numpy as np
import json, tempfile, os
from PIL import Image
from utils.feature_extraction import extract_features

LABEL_NAMES = ['L', 'A', 'B', 'FIRMNESS', 'TA', 'TSS', 'PH', 'REMAINING DAYS TO RIPE']

# ── Load ONNX session ─────────────────────────────────────────────────────────
sess = ort.InferenceSession("model.onnx",
       providers=["CPUExecutionProvider"])

# ── Load label stats (plain numpy, no torch) ──────────────────────────────────
import torch
_stats     = torch.load("label_stats.pt", map_location="cpu")
label_mean = _stats['mean'].numpy()   # (8,)
label_std  = _stats['std'].numpy()    # (8,)
del torch   # free torch after loading stats

# ── Image pre-processing (replaces torchvision transforms) ───────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.resize((224, 224), Image.BILINEAR).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0          # (224,224,3)
    arr = (arr - _MEAN) / _STD                             # normalize
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]          # (1,3,224,224)
    return arr.astype(np.float32)

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(pil_image):
    if pil_image is None:
        return tuple(["—"] * len(LABEL_NAMES))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path)

    try:
        img_arr  = preprocess(pil_image)                              # (1,3,224,224)
        feat_arr = extract_features(tmp_path)[np.newaxis, :]          # (1,190)

        preds = sess.run(["predictions"],
                         {"image": img_arr, "features": feat_arr})[0]  # (1,8)
        preds = preds[0] * label_std + label_mean                      # inverse z-score

        return tuple(f"{v:.4f}" for v in preds.tolist())
    finally:
        os.unlink(tmp_path)

# ── UI ────────────────────────────────────────────────────────────────────────
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
            image_input = gr.Image(type="pil", label="📷 Upload Guava Image", height=320)
            predict_btn = gr.Button("🔍 Predict", variant="primary",
                                    elem_classes="predict-btn")

        with gr.Column(scale=2):
            gr.Markdown("### 🎨 Colour (LAB)")
            with gr.Row():
                out_L = gr.Textbox(label="💡 L  — Lightness",     interactive=False)
                out_A = gr.Textbox(label="🔴 A  — Green / Red",   interactive=False)
                out_B = gr.Textbox(label="🟡 B  — Blue / Yellow", interactive=False)

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
