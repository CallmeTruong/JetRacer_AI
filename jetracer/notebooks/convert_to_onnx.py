#!/usr/bin/env python3
"""
Convert PyTorch ResNet-18 Road Following model (.pth) to ONNX (.onnx)
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torchvision

notebooks_dir = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PyTorch .pth model to ONNX format")
    parser.add_argument("--input", type=str, default=str(notebooks_dir / "road_following_model.pth"),
                        help="Input PyTorch model (.pth)")
    parser.add_argument("--output", type=str, default=str(notebooks_dir / "road_following_model.onnx"),
                        help="Output ONNX model (.onnx)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11 for broad compatibility)")
    return parser.parse_args()

def export_onnx():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"[!] ERROR: Input model '{input_path}' does not exist!")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Loading PyTorch ResNet-18 model from: {input_path}")

    # Load ResNet-18 model structure
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load(input_path, map_location=device))
    model = model.to(device).eval()

    # Create dummy input tensor matching camera resolution (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    print(f"[*] Exporting model to ONNX format: {output_path} (opset {args.opset})...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,  # Fixed shape (1, 3, 224, 224) for optimal ONNX Runtime acceleration
            dynamo=False
        )
        print(f"[+] Successfully converted PyTorch model to ONNX: {output_path}")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    ONNX Model Size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"\n[!] ERROR during ONNX export: {e}")
        print("    Note: PyTorch requires the 'onnx' package to export ONNX files.")
        print("    Run: pip install onnx")

if __name__ == "__main__":
    export_onnx()
