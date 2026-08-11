#!/usr/bin/env python3
"""
Road Following Autonomous Driving Script (Stanley Control + ONNX Model)
Runs inference via ONNX Runtime without PyTorch dependency, commanding JetRacer hardware.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure parent directory is in sys.path for importing Controller.py
notebooks_dir = Path(__file__).resolve().parent
parent_dir = notebooks_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from Controller import StanleyController

def parse_args():
    parser = argparse.ArgumentParser(description="JetRacer Autonomous Road Following (Stanley + ONNX Model)")
    parser.add_argument("--model", type=str, default=str(notebooks_dir / "road_following_model.onnx"),
                        help="Path to ONNX model (.onnx)")
    parser.add_argument("--k", type=float, default=1.2, help="Stanley gain parameter k (default: 1.2)")
    parser.add_argument("--throttle", type=float, default=0.20, help="Base throttle speed (default: 0.20)")
    parser.add_argument("--brake-gain", type=float, default=0.10, help="Brake gain on sharp turns (default: 0.10)")
    parser.add_argument("--bias", type=float, default=0.0, help="Steering bias offset (default: 0.0)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Kalman filter alpha (default: 0.7)")
    parser.add_argument("--config", type=str, default=str(notebooks_dir / "best_pid_config.json"),
                        help="Path to JSON config file if available")
    parser.add_argument("--camera", type=str, default="csi", choices=["csi", "usb"], help="Camera type: csi or usb (default: csi)")
    parser.add_argument("--camera-device", type=int, default=0, help="Camera device index / sensor ID (default: 0)")
    parser.add_argument("--fps", type=int, default=65, help="Camera capture FPS (default: 65)")
    return parser.parse_args()

def preprocess_onnx(image):
    """
    Pure NumPy/OpenCV preprocessing for ONNX inference.
    Replicates torchvision.transforms: ToTensor() + Normalize(mean, std)
    """
    if isinstance(image, np.ndarray):
        # Convert BGR (camera output) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = np.array(image)

    # Scale pixel values from [0, 255] to [0.0, 1.0] and transpose (H, W, C) -> (C, H, W)
    img_float = image_rgb.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)

    # ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_normalized = (img_chw - mean) / std

    # Add batch dimension -> (1, 3, 224, 224)
    return np.expand_dims(img_normalized, axis=0)

def main():
    args = parse_args()

    # 1. Check & Load ONNX Runtime
    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] ERROR: 'onnxruntime' is not installed.")
        print("    Please install it using: pip install onnxruntime  (or onnxruntime-gpu)")
        sys.exit(1)

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"[!] ERROR: ONNX model file '{model_path}' not found!")
        print("    Please run 'python convert_to_onnx.py' first to generate the .onnx file.")
        sys.exit(1)

    print(f"[*] Loading ONNX model from: {model_path}")
    
    # Try GPU providers first if available, fallback gracefully to CPUExecutionProvider
    available_providers = ort.get_available_providers()
    providers_to_try = []
    if 'CUDAExecutionProvider' in available_providers:
        providers_to_try.append('CUDAExecutionProvider')
    providers_to_try.append('CPUExecutionProvider')

    try:
        session = ort.InferenceSession(model_path, providers=providers_to_try)
    except Exception as e:
        print(f"[!] Note: Could not initialize GPU provider ({e}). Falling back to CPUExecutionProvider...")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[+] Loaded ONNX Session with provider: {session.get_providers()}")

    # 2. Load Config JSON if present
    k_stanley = args.k
    base_throttle = args.throttle
    brake_gain = args.brake_gain
    steering_bias = args.bias
    alpha = args.alpha

    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            k_stanley = cfg.get('k', k_stanley)
            base_throttle = cfg.get('base_throttle', base_throttle)
            brake_gain = cfg.get('brake_gain', brake_gain)
            steering_bias = cfg.get('bias', steering_bias)
            alpha = cfg.get('alpha', alpha)
            print(f"[+] Loaded configuration from '{args.config}'")
        except Exception as e:
            print(f"[!] Warning: Could not read config file: {e}")

    print(f"[*] Stanley Parameters -> k: {k_stanley}, Base Throttle: {base_throttle}, Brake Gain: {brake_gain}, Bias: {steering_bias}, Alpha: {alpha}")

    # 3. Initialize JetRacer Hardware & Camera
    try:
        import traceback
        from jetracer.nvidia_racecar import NvidiaRacecar
        
        car = NvidiaRacecar()

        if args.camera.lower() == 'usb':
            from jetcam.usb_camera import USBCamera
            camera = USBCamera(width=224, height=224, capture_device=args.camera_device)
            print(f"[+] USB Camera (device {args.camera_device}) initialized successfully!")
        else:
            from jetcam.csi_camera import CSICamera
            camera = CSICamera(width=224, height=224, capture_device=args.camera_device, capture_fps=args.fps)
            print(f"[+] CSI Camera (sensor_id {args.camera_device}, fps {args.fps}) initialized successfully!")

        camera.running = True

    except ImportError as e:
        print(f"[!] ERROR: JetRacer / JetCam hardware libraries not found ({e}).")
        print("    This script must be executed on Jetson Nano with JetRacer & JetCam installed.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] ERROR initializing hardware/camera:")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print(" TROUBLESHOOTING CAMERA ON JETSON NANO:")
        print(" 1. Reset the CSI camera daemon by running:")
        print("    sudo systemctl restart nvargus-daemon")
        print(" 2. Close Jupyter Notebook kernels or any process using the camera.")
        print(" 3. If using a USB camera instead of CSI, run with:")
        print("    python3 road_following_stanley_onnx.py --camera usb")
        print("="*60 + "\n")
        sys.exit(1)

    # 4. Initialize Stanley Controller
    stanley = StanleyController()
    stanley.reset()

    print("\n=======================================================")
    print("   AUTONOMOUS DRIVING STARTED (ONNX + Stanley Control) ")
    print("   Press Ctrl+C to stop the car and exit safely.       ")
    print("=======================================================\n")

    try:
        while True:
            # Capture camera frame
            image = camera.value
            if image is None:
                time.sleep(0.01)
                continue

            # Preprocess frame for ONNX model
            input_tensor = preprocess_onnx(image)

            # ONNX Inference
            outputs = session.run([output_name], {input_name: input_tensor})
            output_data = outputs[0].flatten()

            raw_x = float(output_data[0])

            # Update Stanley Controller
            steering, dyn_throttle = stanley.update(
                raw_x=raw_x,
                k=k_stanley,
                base_throttle=base_throttle,
                brake_gain=brake_gain,
                bias=steering_bias,
                alpha=alpha
            )

            # Command actuators
            car.steering = steering
            car.throttle = dyn_throttle

            # Console status display
            sys.stdout.write(f"\r[ONNX Live] Target X: {raw_x:+.3f} | Smoothed X: {stanley.smoothed_x:+.3f} | Steering: {steering:+.3f} | Throttle: {dyn_throttle:.3f}")
            sys.stdout.flush()

            time.sleep(0.02) # ~50 FPS loop

    except KeyboardInterrupt:
        print("\n\n[*] KeyboardInterrupt detected! Stopping car safely...")
    except Exception as e:
        print(f"\n\n[!] Error during ONNX drive loop: {e}")
    finally:
        # Safe stop
        car.throttle = 0.0
        car.steering = 0.0
        print("[+] Car stopped safely. Exiting.")

if __name__ == "__main__":
    main()
