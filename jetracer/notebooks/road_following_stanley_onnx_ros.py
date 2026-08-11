#!/usr/bin/env python3
"""
Road Following Autonomous Driving Script (Stanley Control + ONNX Model) via ROS Topic
Subscribes to ROS Camera Topic (/csi_cam_0/image_raw), avoiding camera hardware conflicts.
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
    parser = argparse.ArgumentParser(description="JetRacer Autonomous Road Following (Stanley + ONNX Model via ROS)")
    parser.add_argument("--model", type=str, default=str(notebooks_dir / "road_following_model.onnx"),
                        help="Path to ONNX model (.onnx)")
    parser.add_argument("--topic", type=str, default="/csi_cam_0/image_raw",
                        help="ROS Image topic name (default: /csi_cam_0/image_raw)")
    parser.add_argument("--k", type=float, default=1.2, help="Stanley gain parameter k (default: 1.2)")
    parser.add_argument("--throttle", type=float, default=0.20, help="Base throttle speed (default: 0.20)")
    parser.add_argument("--brake-gain", type=float, default=0.10, help="Brake gain on sharp turns (default: 0.10)")
    parser.add_argument("--bias", type=float, default=0.0, help="Steering bias offset (default: 0.0)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Kalman filter alpha (default: 0.7)")
    parser.add_argument("--config", type=str, default=str(notebooks_dir / "best_pid_config.json"),
                        help="Path to JSON config file if available")
    return parser.parse_args()

def preprocess_onnx(image):
    if isinstance(image, np.ndarray):
        # Resize to 224x224 if needed
        if image.shape[0] != 224 or image.shape[1] != 224:
            image = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = np.array(image)

    img_float = image_rgb.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_normalized = (img_chw - mean) / std

    return np.expand_dims(img_normalized, axis=0)

class JetRacerROSOnnxRunner:
    def __init__(self, args, session, input_name, output_name, car, stanley):
        self.args = args
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.car = car
        self.stanley = stanley

        self.k_stanley = args.k
        self.base_throttle = args.throttle
        self.brake_gain = args.brake_gain
        self.steering_bias = args.bias
        self.alpha = args.alpha

        if os.path.exists(args.config):
            try:
                with open(args.config, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.k_stanley = cfg.get('k', self.k_stanley)
                self.base_throttle = cfg.get('base_throttle', self.base_throttle)
                self.brake_gain = cfg.get('brake_gain', self.brake_gain)
                self.steering_bias = cfg.get('bias', self.steering_bias)
                self.alpha = cfg.get('alpha', self.alpha)
                print(f"[+] Loaded configuration from '{args.config}'")
            except Exception as e:
                print(f"[!] Warning: Could not read config file: {e}")

        self.first_frame_received = False

    def ros_image_to_cv2(self, msg):
        """Pure NumPy ROS Image decoding to bypass ROS Melodic Python 3 cv_bridge C++ Boost issues."""
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ['rgb8', 'rgb8']:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'rgba8':
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == 'bgra8':
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        return im

    def image_callback(self, msg):
        try:
            if not self.first_frame_received:
                self.first_frame_received = True
                print("[+] First ROS camera frame received! Autonomous driving active.\n")

            cv_image = self.ros_image_to_cv2(msg)

            input_tensor = preprocess_onnx(cv_image)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            output_data = outputs[0].flatten()
            raw_x = float(output_data[0])

            steering, dyn_throttle = self.stanley.update(
                raw_x=raw_x,
                k=self.k_stanley,
                base_throttle=self.base_throttle,
                brake_gain=self.brake_gain,
                bias=self.steering_bias,
                alpha=self.alpha
            )

            self.car.steering = steering
            self.car.throttle = dyn_throttle

            sys.stdout.write(f"\r[ROS ONNX Live] Target X: {raw_x:+.3f} | Smoothed X: {self.stanley.smoothed_x:+.3f} | Steering: {steering:+.3f} | Throttle: {dyn_throttle:.3f}")
            sys.stdout.flush()

        except Exception as e:
            print(f"\n[!] Error in image_callback: {e}")

def main():
    args = parse_args()

    # 1. ROS Setup
    try:
        import rospy
        from sensor_msgs.msg import Image
        rospy.init_node('road_following_stanley_onnx', anonymous=True)
    except ImportError:
        print("[!] ERROR: ROS ('rospy') is not installed or sourced.")
        print("    Please run: source /opt/ros/melodic/setup.bash (or catkin workspace setup.bash)")
        sys.exit(1)

    # 2. ONNX Session
    import onnxruntime as ort
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"[!] ERROR: ONNX model '{model_path}' not found!")
        sys.exit(1)

    print(f"[*] Loading ONNX model from: {model_path}")
    available_providers = ort.get_available_providers()
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else []
    providers.append('CPUExecutionProvider')

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 3. Hardware & Controller
    from jetracer.nvidia_racecar import NvidiaRacecar
    car = NvidiaRacecar()
    stanley = StanleyController()
    stanley.reset()

    runner = JetRacerROSOnnxRunner(args, session, input_name, output_name, car, stanley)

    print(f"[*] Subscribing to ROS Image Topic: {args.topic}")
    rospy.Subscriber(args.topic, Image, runner.image_callback)

    print("\n=======================================================")
    print("   AUTONOMOUS DRIVING STARTED (ROS Topic + ONNX)       ")
    print("   Waiting for frames from topic: " + args.topic)
    print("   (Ensure launch_camera.sh is running in terminal 1)   ")
    print("   Press Ctrl+C to stop the car and exit safely.       ")
    print("=======================================================\n")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("\n[*] Stopping car safely...")
    finally:
        car.throttle = 0.0
        car.steering = 0.0
        print("[+] Car stopped safely. Exiting.")

if __name__ == "__main__":
    main()
