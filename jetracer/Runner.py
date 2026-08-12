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
        video_path=None,
        video_fps=20.0,
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

        self.video_path = video_path
        self.video_fps = video_fps
        self.video_writer = None

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

    def stop(self):
        """Stop driving safely and release video writer if active."""
        self.running = False
        if self.car is not None:
            self.car.throttle = 0.0
            self.car.steering = 0.0
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"\n[+] Saved recorded video to: {self.video_path}")

    def ros_image_to_cv2(self, msg):
        """Pure NumPy ROS Image decoding to bypass ROS Melodic Python 3 cv_bridge C++ Boost issues."""
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ['rgb8', 'rgb8']:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'rgba8':
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == 'bgra8':
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        
        # Standardize to 224x224 so target points and overlay map 100% accurately
        if im.shape[0] != 224 or im.shape[1] != 224:
            im = cv2.resize(im, (224, 224))
        return im


    def image_callback(self, msg):
        if not self.running:
            return

        # Non-blocking lock acquire: drop frame immediately if previous inference is still running
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
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

            # Video Recording
            if self.video_path is not None:
                h, w = cv_image.shape[:2]
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.video_fps, (w, h))

                annotated = cv_image.copy()
                px = int(w * (self.stanley.smoothed_x / 2.0 + 0.5))
                py = int(h * (raw_y / 2.0 + 0.5)) if raw_y != 0.0 else int(h * 0.5)
                cv2.circle(annotated, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(annotated, f"Steer:{steering:+.2f} Thr:{dyn_throttle:.2f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.video_writer.write(annotated)

            # Custom Frame Callback (e.g. In-place HTML Jupyter display)
            if self.on_frame is not None:
                self.on_frame(cv_image, raw_x, raw_y, self.stanley.smoothed_x, steering, dyn_throttle)

        except Exception as e:
            print(f"\n[!] Error in image_callback: {e}")
        finally:
            try:
                self._lock.release()
            except RuntimeError:
                pass


class JetRacerROSPthRunner:
    def __init__(
        self,
        model,
        device,
        car,
        stanley,
        k=2.5,
        throttle=0.6,
        brake_gain=0.10,
        bias=0.0,
        alpha=0.4,
        video_path=None,
        video_fps=20.0,
        on_frame=None
    ):
        self.model = model
        self.device = device
        self.car = car
        self.stanley = stanley

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
        self.first_frame_received = False

        self._lock = threading.Lock()

    def _eval_param(self, param):
        return param() if callable(param) else param

    def stop(self):
        self.running = False
        if self.car:
            self.car.steering = 0.0
            self.car.throttle = 0.0
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def ros_image_to_cv2(self, msg):
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding in ['rgb8', 'rgb8']:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'rgba8':
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        elif msg.encoding == 'bgra8':
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        
        if im.shape[0] != 224 or im.shape[1] != 224:
            im = cv2.resize(im, (224, 224))
        return im

    def image_callback(self, msg):
        if not self.running:
            return

        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return

        try:
            if not self.first_frame_received:
                self.first_frame_received = True
                print("[+] First ROS camera frame received! PyTorch Autonomous driving active.\n")

            cv_image = self.ros_image_to_cv2(msg)

            # PyTorch Preprocessing & Inference
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            img_chw = img_float.transpose(2, 0, 1)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            img_norm = (img_chw - mean) / std
            input_tensor = np.expand_dims(img_norm, axis=0)

            import torch
            with torch.no_grad():
                img_t = torch.from_numpy(input_tensor).to(self.device)
                output_data = self.model(img_t).cpu().numpy().flatten()

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

            sys.stdout.write(f"\r[ROS PyTorch Live] Target X: {raw_x:+.3f} | Smoothed X: {self.stanley.smoothed_x:+.3f} | Steering: {steering:+.3f} | Throttle: {dyn_throttle:.3f}")
            sys.stdout.flush()

            # Video Recording
            if self.video_path is not None:
                h, w = cv_image.shape[:2]
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.video_fps, (w, h))

                annotated = cv_image.copy()
                px = int(w * (self.stanley.smoothed_x / 2.0 + 0.5))
                py = int(h * (raw_y / 2.0 + 0.5)) if raw_y != 0.0 else int(h * 0.5)
                cv2.circle(annotated, (px, py), 8, (0, 255, 0), -1)
                cv2.putText(annotated, f"Steer:{steering:+.2f} Thr:{dyn_throttle:.2f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                self.video_writer.write(annotated)

            if self.on_frame is not None:
                self.on_frame(cv_image, raw_x, raw_y, self.stanley.smoothed_x, steering, dyn_throttle)

        except Exception as e:
            print(f"\n[!] Error in image_callback: {e}")
        finally:
            try:
                self._lock.release()
            except RuntimeError:
                pass







