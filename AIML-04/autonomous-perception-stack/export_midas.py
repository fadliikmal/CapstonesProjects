import os
import torch
import onnxruntime as ort

# Patch nn.Unflatten so dynamic Tensor dimensions from MiDaS ViT backbone
# are converted to Python ints during ONNX export
_orig_unflatten_init = torch.nn.Unflatten.__init__


def _patched_unflatten_init(self, dim, unflattened_size):
    if isinstance(unflattened_size, (tuple, list)):
        unflattened_size = tuple(
            int(s.item()) if isinstance(s, torch.Tensor) else int(s)
            for s in unflattened_size
        )
    _orig_unflatten_init(self, dim, unflattened_size)


torch.nn.Unflatten.__init__ = _patched_unflatten_init


def export_dpt_hybrid_to_onnx(output_path: str = "models/midas_dpt_hybrid.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("[INFO] Loading pre-trained MiDaS DPT_Hybrid weights...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
    midas.eval()

    dummy_input = torch.randn(1, 3, 384, 384, dtype=torch.float32)

    print(f"[INFO] Exporting DPT_Hybrid to ONNX at '{output_path}'...")
    export_kwargs = dict(
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["depth"],
    )
    try:
        torch.onnx.export(midas, dummy_input, output_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(midas, dummy_input, output_path, **export_kwargs)

    # Verify exported ONNX model with ONNX Runtime CPU
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out = sess.run(["depth"], {"input": dummy_input.numpy()})[0]
    print(f"[SUCCESS] Verified '{output_path}' | Output shape: {out.shape}")


if __name__ == "__main__":
    export_dpt_hybrid_to_onnx()