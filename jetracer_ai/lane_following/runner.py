"""
JetRacer Stanley ONNX ROS Runner for Lane Following.
"""
import os
import json
import threading
import cv2
import numpy as np

from jetracer_ai.utils.image_utils import preprocess_onnx
from jetracer_ai.lane_following.controller import StanleyController


class JetRacerStanleyRunner:
    """ROS execution runner for real-time ONNX lane following with Stanley steering control."""
    def __init__(
        self,
        session,
        input_name,
        output_name,
        car=None,
        stanley=None,
        k=2.5,
        throttle=0.6,
        brake_gain=0.10,
        bias=0.0,
        alpha=0.4,
        config_path=None,
        video_path=None,
        video_fps=20.0,
        on_frame=None,
        lidar_processor=None
    ):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.car = car
        self.stanley = stanley or StanleyController()
        self.lidar_processor = lidar_processor

        self.k_stanley = k
        self.base_throttle = throttle
        self.brake_gain = brake_gain
        self.steering_bias = bias
        self.alpha = alpha
        self.on_frame = on_frame
        self.running = True

        self.video_path = video_path
        self.video_fps = video_fps
        self.video_writer = None
        self._lock = threading.Lock()

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.k_stanley = cfg.get('k', self.k_stanley)
                self.base_throttle = cfg.get('base_throttle', self.base_throttle)
                self.brake_gain = cfg.get('brake_gain', self.brake_gain)
                self.steering_bias = cfg.get('bias', self.steering_bias)
                self.alpha = cfg.get('alpha', self.alpha)
            except Exception as e:
                print(f"[!] Warning: Could not read config file: {e}")

        self.first_frame_received = False

    def update_params(self, k=None, throttle=None, brake_gain=None, bias=None, alpha=None):
        """Update controller parameters dynamically at runtime."""
        if k is not None: self.k_stanley = k
        if throttle is not None: self.base_throttle = throttle
        if brake_gain is not None: self.brake_gain = brake_gain
        if bias is not None: self.steering_bias = bias
        if alpha is not None: self.alpha = alpha

    def stop(self):
        """Stop driving safely and release video writer if active."""
        self.running = False
        if self.car is not None:
            self.car.throttle = 0.0
            self.car.steering = 0.0
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def process_frame(self, image):
        """Process OpenCV BGR image and return steering & throttle."""
        if image is None:
            return 0.0, 0.0

        img_tensor = preprocess_onnx(image)
        outputs = self.session.run([self.output_name], {self.input_name: img_tensor})
        raw_x = float(outputs[0][0][0])

        lidar_offset = 0.0
        if self.lidar_processor is not None:
            lidar_offset = self.lidar_processor.get_evasive_offset()

        steering, dyn_throttle = self.stanley.update(
            raw_x=raw_x,
            k=self.k_stanley,
            base_throttle=self.base_throttle,
            brake_gain=self.brake_gain,
            bias=self.steering_bias,
            alpha=self.alpha,
            lidar_offset=lidar_offset
        )

        if self.car is not None:
            self.car.steering = steering
            self.car.throttle = dyn_throttle

        return steering, dyn_throttle
