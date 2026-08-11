import os
import sys
import cv2
import numpy as np
import json
import threading

try:
    from jetracer.utils import preprocess_onnx
except ImportError:
    from utils import preprocess_onnx

class JetRacerROSOnnxRunner:
    def __init__(
        self,
        session,
        input_name,
        output_name,
        car,
        stanley,
        k=2.5,
        throttle=0.6,
        brake_gain=0.10,
        bias=0.0,
        alpha=0.4,
        config_path=None,
        on_frame=None
    ):
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.car = car
        self.stanley = stanley

        self.k_stanley = k
        self.base_throttle = throttle
        self.brake_gain = brake_gain
        self.steering_bias = bias
        self.alpha = alpha
        self.on_frame = on_frame
        self.running = True

        self._lock = threading.Lock()  # Non-blocking lock to drop old frames if inference is busy

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.k_stanley = cfg.get('k', self.k_stanley)
                self.base_throttle = cfg.get('base_throttle', self.base_throttle)
                self.brake_gain = cfg.get('brake_gain', self.brake_gain)
                self.steering_bias = cfg.get('bias', self.steering_bias)
                self.alpha = cfg.get('alpha', self.alpha)
                print(f"[+] Loaded configuration from '{config_path}'")
            except Exception as e:
                print(f"[!] Warning: Could not read config file: {e}")

        self.first_frame_received = False

    def _eval_param(self, val):
        return val() if callable(val) else val

    def update_params(self, k=None, throttle=None, brake_gain=None, bias=None, alpha=None):
        """Update controller parameters dynamically at runtime."""
        if k is not None: self.k_stanley = k
        if throttle is not None: self.base_throttle = throttle
        if brake_gain is not None: self.brake_gain = brake_gain
        if bias is not None: self.steering_bias = bias
        if alpha is not None: self.alpha = alpha

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
        if not self.running:
            return

        # Non-blocking lock acquire: drop frame immediately if previous inference is still running
        if not self._lock.acquire(blocking=False):
            return

        try:
            if not self.first_frame_received:
                self.first_frame_received = True
                print("[+] First ROS camera frame received! Autonomous driving active.\n")

            cv_image = self.ros_image_to_cv2(msg)

            input_tensor = preprocess_onnx(cv_image)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            output_data = outputs[0].flatten()
            raw_x = float(output_data[0])
            raw_y = float(output_data[1]) if len(output_data) > 1 else 0.0

            k_val = self._eval_param(self.k_stanley)
            throttle_val = self._eval_param(self.base_throttle)
            brake_val = self._eval_param(self.brake_gain)
            bias_val = self._eval_param(self.steering_bias)
            alpha_val = self._eval_param(self.alpha)

            steering, dyn_throttle = self.stanley.update(
                raw_x=raw_x,
                k=k_val,
                base_throttle=throttle_val,
                brake_gain=brake_val,
                bias=bias_val,
                alpha=alpha_val
            )

            self.car.steering = steering
            self.car.throttle = dyn_throttle

            sys.stdout.write(f"\r[ROS ONNX Live] Target X: {raw_x:+.3f} | Smoothed X: {self.stanley.smoothed_x:+.3f} | Steering: {steering:+.3f} | Throttle: {dyn_throttle:.3f}")
            sys.stdout.flush()

            if self.on_frame is not None:
                self.on_frame(cv_image, raw_x, raw_y, self.stanley.smoothed_x, steering, dyn_throttle)

        except Exception as e:
            print(f"\n[!] Error in image_callback: {e}")
        finally:
            self._lock.release()



