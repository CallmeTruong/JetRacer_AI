import os
import sys
import cv2
import numpy as np
import threading
from jetracer.urban.config import COMMAND_TO_INDEX

class UrbanAutonomousRunner:
    """
    Multi-threaded Real-time Execution Runner for Urban Autonomous Driving.
    Connects: ROS Camera + Conditioned Lane Model + Object Detector + FSM Planner + Pure Pursuit Controller + Hardware.
    """
    def __init__(
        self,
        lane_session,
        detector,
        fsm,
        controller,
        car,
        route_command='STRAIGHT',
        lookahead=0.45,
        throttle=0.20,
        video_path=None,
        on_frame=None
    ):
        self.lane_session = lane_session
        self.detector = detector
        self.fsm = fsm
        self.controller = controller
        self.car = car
        
        self.route_command = route_command
        self.lookahead = lookahead
        self.throttle = throttle
        self.on_frame = on_frame
        self.running = True
        
        self.video_path = video_path
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
                print("[+] Urban Autonomous System Active: First Camera Frame Received!\n")

            cv_image = self.ros_image_to_cv2(msg)

            # 1. Run Model 2: Object Detection (detect green_light, red_light, stop_sign, crosswalk, etc.)
            detections = []
            if self.detector is not None:
                detections = self.detector.detect(cv_image)

            # 2. Update FSM Planner State
            current_cmd = self._eval_param(self.route_command)
            self.fsm.set_route_command(current_cmd)
            is_stopped, active_cmd, fsm_status = self.fsm.update(detections)

            # 3. Run Model 1: Conditioned Lane Trajectory Prediction
            # Preprocess image
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_norm = (img_rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img_chw = img_norm.transpose(2, 0, 1)
            img_input = np.expand_dims(img_chw, axis=0).astype(np.float32)
            
            cmd_idx = np.array([COMMAND_TO_INDEX.get(active_cmd, 1)], dtype=np.int64)

            # Model 1 Inference
            img_in_name = self.lane_session.get_inputs()[0].name
            cmd_in_name = self.lane_session.get_inputs()[1].name if len(self.lane_session.get_inputs()) > 1 else None
            out_name = self.lane_session.get_outputs()[0].name
            
            if cmd_in_name:
                outs = self.lane_session.run([out_name], {img_in_name: img_input, cmd_in_name: cmd_idx})
            else:
                outs = self.lane_session.run([out_name], {img_in_name: img_input})
                
            raw_waypoints = outs[0].reshape(-1, 2) # (5, 2)

            # 4. Pure Pursuit Trajectory Tracking & Control
            Ld_val = self._eval_param(self.lookahead)
            base_thr_val = self._eval_param(self.throttle)

            steering, dyn_throttle, target_point = self.controller.update(
                waypoints=raw_waypoints,
                lookahead_distance=Ld_val,
                base_throttle=base_thr_val
            )

            # Enforce FSM Stop Decision
            if is_stopped:
                dyn_throttle = 0.0

            # 5. Apply Control Commands to JetRacer Hardware
            self.car.steering = steering
            self.car.throttle = dyn_throttle

            sys.stdout.write(f"\r[Urban Auto] FSM: {self.fsm.state:<16} | Route: {active_cmd:<8} | Steer: {steering:+.2f} | Thr: {dyn_throttle:.2f}")
            sys.stdout.flush()

            # 6. Custom Frame Callback (Live ipywidgets UI update)
            if self.on_frame is not None:
                self.on_frame(cv_image, raw_waypoints, detections, is_stopped, active_cmd, fsm_status, steering, dyn_throttle)

        except Exception as e:
            print(f"\n[!] Error in Urban Runner image_callback: {e}")
        finally:
            try:
                self._lock.release()
            except RuntimeError:
                pass
