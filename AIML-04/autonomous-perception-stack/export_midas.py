# Install dependencies if running on a fresh Google Colab runtime
# !pip install -q timm onnx onnxruntime

import os
import shutil
import torch
import onnx

# 1. Load pre-trained MiDaS_small from PyTorch Hub
model_type = "MiDaS_small"
print(f"[INFO] Loading {model_type} from PyTorch Hub...")
midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
midas.eval()

# Export on CPU for maximum ONNX graph compatibility
device = torch.device("cpu")
midas.to(device)

# 2. Create dummy input matching MiDaS_small native resolution (1, 3, 256, 256)
dummy_input = torch.randn(1, 3, 256, 256, device=device)

# 3. Export PyTorch model to ONNX format
os.makedirs("models", exist_ok=True)
onnx_output_path = "models/midas_small.onnx"

print(f"[INFO] Exporting {model_type} to {onnx_output_path}...")
torch.onnx.export(
    midas,
    dummy_input,
    onnx_output_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["depth"],
)

# 4. Validate the exported ONNX graph
onnx_model = onnx.load(onnx_output_path)
onnx.checker.check_model(onnx_model)
print(f"[SUCCESS] ONNX model verified and saved at: {onnx_output_path}")

# 5. Optional: Auto-backup to Google Drive if executed inside Google Colab
drive_export_dir = "/content/drive/MyDrive/Autonomous_Perception_Models/kitti_perception/exported_models"
if os.path.exists("/content/drive/MyDrive"):
    os.makedirs(drive_export_dir, exist_ok=True)
    drive_dst = os.path.join(drive_export_dir, "midas_small.onnx")
    shutil.copy(onnx_output_path, drive_dst)
    print(f"[SUCCESS] Backup copied to Google Drive: {drive_dst}")