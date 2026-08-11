#!/usr/bin/env python3
"""
Road Following Autonomous Driving Script (Stanley Control + ONNX Model) via ROS Topic
Subscribes to ROS Camera Topic (/csi_cam_0/image_raw), avoiding camera hardware conflicts.
"""

import os
import sys
import time
from pathlib import Path

# Ensure parent directory is in sys.path for importing Controller.py, Runner.py, utils.py
notebooks_dir = Path(__file__).resolve().parent
parent_dir = notebooks_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from Controller import StanleyController
from Runner import JetRacerROSOnnxRunner

# Default Parameters (No argparse required)
MODEL_PATH = str(notebooks_dir / "road_following_model.onnx")
TOPIC = "/csi_cam_0/image_raw"
CONFIG_PATH = str(notebooks_dir / "best_pid_config.json")
K_STANLEY = 2.5
THROTTLE = 0.6
BRAKE_GAIN = 0.10
BIAS = 0.0
ALPHA = 0.4

def main():
    # 1. ROS Setup
    try:
        import rospy
        from sensor_msgs.msg import Image
        rospy.init_node('road_following_stanley_onnx', anonymous=True)
    except ImportError:
        print("[!] ERROR: ROS ('rospy') is not installed or sourced.")
        print("    Please run: source /opt/ros/melodic/setup.bash (or catkin workspace setup.bash)")
        sys.exit(1)

    # 2. ONNX Session Setup
    import onnxruntime as ort
    if not os.path.exists(MODEL_PATH):
        print(f"[!] ERROR: ONNX model '{MODEL_PATH}' not found!")
        sys.exit(1)

    print(f"[*] Loading ONNX model from: {MODEL_PATH}")
    available_providers = ort.get_available_providers()
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else []
    providers.append('CPUExecutionProvider')

    try:
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
    except Exception:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 3. Hardware & Controller Setup
    from jetracer.nvidia_racecar import NvidiaRacecar
    car = NvidiaRacecar()
    stanley = StanleyController()
    stanley.reset()

    # 4. Instantiate ROS ONNX Runner
    runner = JetRacerROSOnnxRunner(
        session=session,
        input_name=input_name,
        output_name=output_name,
        car=car,
        stanley=stanley,
        k=K_STANLEY,
        throttle=THROTTLE,
        brake_gain=BRAKE_GAIN,
        bias=BIAS,
        alpha=ALPHA,
        config_path=CONFIG_PATH
    )

    print(f"[*] Subscribing to ROS Image Topic: {TOPIC}")
    rospy.Subscriber(TOPIC, Image, runner.image_callback, queue_size=1, buff_size=2**24)

    print("\n=======================================================")
    print("   AUTONOMOUS DRIVING STARTED (ROS Topic + ONNX)       ")
    print("   Waiting for frames from topic: " + TOPIC)
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

