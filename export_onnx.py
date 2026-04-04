"""
export_onnx.py — converts model.pth → model.onnx
Run once: python export_onnx.py
"""
import torch
from models.model import GuavaNet

model = GuavaNet(handcrafted_dim=190, num_outputs=8)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

dummy_img  = torch.randn(1, 3, 224, 224)
dummy_feat = torch.randn(1, 190)

torch.onnx.export(
    model,
    (dummy_img, dummy_feat),
    "model.onnx",
    input_names=["image", "features"],
    output_names=["predictions"],
    dynamic_axes={
        "image":       {0: "batch"},
        "features":    {0: "batch"},
        "predictions": {0: "batch"},
    },
    opset_version=17,
)
print("Exported → model.onnx")
