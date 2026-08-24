"""
Centralized Settings Configuration for JetRacer AI Platform.
"""
import os
import cv2

# ============================================================
# BASE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

# ============================================================
# ROS TOPICS
# ============================================================
ROS_TOPIC_CAM = "/csi_cam_0/image_raw"
ROS_TOPIC_SCAN = "/scan"

# ============================================================
# CAMERA & VIDEO RECORDING
# ============================================================
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
VIDEO_FPS = 20.0
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
SMART_CITY_VIDEO_OUTPUT = "smart_city_output.mp4"
SPEED_TRACK_VIDEO_OUTPUT = "speed_track_output.mp4"

# ============================================================
# HARDWARE & CONTROL GAINS
# ============================================================
DEFAULT_CONTROL_CONFIG = {
    # Throttle
    "BASE_THROTTLE": 0.20,
    "TURN_THROTTLE": 0.15,
    "MAX_THROTTLE": 0.40,

    # Steering
    "STEERING_GAIN": 0.8,
    "MAX_STEERING": 1.0,
    "STEERING_OFFSET": 0.0,

    # Turning time
    "TURN_DURATION_90_DEG": 1.5,
    "STEERING_VALUE_FOR_TURN": 0.7,

    # PID Controller
    "PID_KP": 0.5,
    "PID_KI": 0.0,
    "PID_KD": 0.1,

    # safe zone
    "SAFE_ZONE_PERCENT": 0.3,
}
