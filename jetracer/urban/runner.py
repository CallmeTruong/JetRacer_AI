import os
import sys
import time
import cv2
import numpy as np
import threading

try:
    from jetracer.utils import preprocess_onnx
    from jetracer.urban.config import COMMAND_TO_INDEX
except ImportError:
    from utils import preprocess_onnx
    from .config import COMMAND_TO_INDEX


class UrbanAutonomousRunner:
    """
    Complete Real-Time Execution Manager for Urban Autonomous Driving.

    Kinematic Rotational Evade FSM (NO IMU / NO ENCODER VERSION):
      - When BLOCKED, reverses with steer direction oriented so the FRONT NOSE / CAMERA
        rotates towards the active `route_command`.
      - Lock is released IMMEDIATELY after hardware dispatch for fast camera streaming.

    FIXES vs previous version (no IMU available, so all fixes are open-loop /
    heuristic -- there is no ground-truth heading feedback to close the loop with):

      1. EMA-smoothed `prob_blocked` + hysteresis (enter/release thresholds) to stop
         the drive/reverse "fighting" caused by raw classifier noise near the boundary.

      2. Evade direction is now computed from the LAST KNOWN-GOOD waypoints (captured
         while the path was confidently free), not from the current frame's waypoints.
         The current frame is looking at the obstacle itself, so its trajectory
         prediction is out-of-distribution / unreliable -- using stale-but-trustworthy
         data avoids picking the wrong escape direction.

      3. Evade steering magnitude reduced from full ±1.0 to a configurable
         `evade_steer_magnitude` (default 0.75). Since there's no feedback to detect
         "have we rotated enough yet", using full-lock steering for a fixed 1.8s makes
         the overshoot/undershoot error larger. A gentler angle for the same duration
         produces a smaller, more repeatable rotation, corrected further by repeated
         evade cycles if still blocked (rather than one large kick that can send the
         car far off the original route in a single maneuver).

      4. CHECK_FORWARD no longer drives blindly straight (steering=0.0). It now blends
         in the LIVE trajectory-model steering (scaled down) so that as soon as the
         path is clear again, the car starts correcting back toward the lane center
         instead of continuing in whatever direction the open-loop reverse left it
         pointed at. This is the main fix for drifting far from the original route --
         previously the car had no way to re-align with the path after an evade.

      5. `max_evade_attempts`: if the evade cycle repeats more times than this without
         clearing, the runner enters a `HALTED` failsafe state (full stop, no more
         automatic maneuvers) instead of endlessly reversing/turning and drifting
         further from the route. Operator must toggle RUN off/on to clear it.
    """
    def __init__(
        self,
        lane_session,
        safety_session=None,
        detector=None,
        fsm=None,
        controller=None,
        car=None,
        route_command='STRAIGHT',
        lookahead=0.40,
        throttle=0.18,
        blocked_threshold=0.50,
        blocked_release_margin=0.15,     # hysteresis gap below blocked_threshold to clear the block
        prob_blocked_ema_alpha=0.35,     # EMA smoothing factor for the safety-model probability
        evade_steer_magnitude=0.75,      # reduced from full ±1.0 -- gentler, more repeatable rotation
        evade_reverse_duration=1.4,      # seconds of reverse+turn per evade cycle (was 1.8)
        evade_pause_duration=0.3,
        evade_check_duration=1.0,
        max_evade_attempts=4,            # stop retrying forever; enter HALTED instead
        on_frame=None
    ):
        self.lane_session       = lane_session
        self.safety_session     = safety_session
        self.detector           = detector
        self.fsm                = fsm
        self.controller         = controller
        self.car                = car

        self.route_command      = route_command
        self.lookahead          = lookahead
        self.throttle           = throttle
        self.blocked_threshold  = blocked_threshold
        self.blocked_release_margin = blocked_release_margin
        self.prob_blocked_ema_alpha  = prob_blocked_ema_alpha

        self.evade_steer_magnitude  = evade_steer_magnitude
        self.evade_reverse_duration = evade_reverse_duration
        self.evade_pause_duration   = evade_pause_duration
        self.evade_check_duration   = evade_check_duration
        self.max_evade_attempts     = max_evade_attempts

        self.on_frame            = on_frame
        self.running             = False

        self.first_frame_received = False
        self._lock = threading.Lock()

        # ── Rotational Evade FSM State Variables ──────────────────────────────
        self.confirm_frames        = 3
        self.blocked_frame_count   = 0
        self.evade_state           = 'DRIVE'   # 'DRIVE','REVERSE_TURNING','PAUSE','CHECK_FORWARD','HALTED'
        self.evade_start_time      = 0.0
        self.evade_steer_dir       = -1.0
        self.evade_attempts        = 0

        # ── Smoothing / memory state ───────────────────────────────────────────
        self._prob_blocked_smoothed = 0.0
        self.last_valid_wps         = None   # waypoints captured while confidently FREE

    def _eval_param(self, param):
        return param() if callable(param) else param

    def reset_evade_fsm(self):
        self.evade_state            = 'DRIVE'
        self.blocked_frame_count    = 0
        self.evade_start_time       = 0.0
        self.evade_attempts         = 0
        self._prob_blocked_smoothed = 0.0

    def stop(self):
        self.running = False
        self.reset_evade_fsm()
        if self.car is not None:
            self.car.steering = 0.0
            self.car.throttle = 0.0

    def ros_image_to_cv2(self, msg):
        """Pure NumPy ROS Image decoder matching Runner.py."""
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        enc = str(msg.encoding).lower()
        if enc in ['rgb8']:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        elif enc in ['rgba8']:
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        elif enc in ['bgra8']:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        if im.shape[0] != 224 or im.shape[1] != 224:
            im = cv2.resize(im, (224, 224))
        return im

    def _compute_evade_direction(self, active_cmd, reference_wps):
        """Decide which way to steer while reversing so the nose rotates toward
        the intended direction. Uses `reference_wps` (ideally the LAST KNOWN-GOOD
        waypoints, not the current -- possibly obstacle-contaminated -- frame)."""
        if active_cmd == 'LEFT':
            return +1.0   # reverse steer RIGHT -> nose rotates LEFT
        elif active_cmd == 'RIGHT':
            return -1.0   # reverse steer LEFT -> nose rotates RIGHT
        else:  # 'STRAIGHT'
            if reference_wps is not None and len(reference_wps) > 0:
                cx = float(np.mean(reference_wps[:, 0]))
                return +1.0 if cx < 112.0 else -1.0
            return +1.0

    def image_callback(self, msg):
        if not self.running:
            return

        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return

        try:
            if not self.first_frame_received:
                self.first_frame_received = True
                print("[+] First ROS camera frame received! Urban Autonomous Driving active.\n")

            cv_image = self.ros_image_to_cv2(msg)

            # ── 1. Object Detector (YOLO) ─────────────────────────────────────
            detections = []
            if self.detector is not None:
                try:
                    detections = self.detector.detect(cv_image)
                except Exception as e:
                    print(f"[*] Detector notice: {e}")

            # ── 2. FSM Planner (Traffic Lights, Stop Signs, Crosswalk) ────────
            current_cmd = self._eval_param(self.route_command)
            is_fsm_stop = False
            active_cmd  = current_cmd
            fsm_status  = "DRIVE"
            if self.fsm is not None:
                try:
                    self.fsm.set_route_command(current_cmd)
                    is_fsm_stop, active_cmd, fsm_status = self.fsm.update(detections)
                except Exception as e:
                    print(f"[*] FSM notice: {e}")

            # ── 3. Preprocess Image ───────────────────────────────────────────
            img_input = preprocess_onnx(cv_image)
            cmd_idx   = np.array([COMMAND_TO_INDEX.get(active_cmd, 1)], dtype=np.int64)

            # ── 4a. Trajectory Model Inference ────────────────────────────────
            traj_inputs = self.lane_session.get_inputs()
            if len(traj_inputs) > 1:
                traj_outs = self.lane_session.run(
                    None, {traj_inputs[0].name: img_input, traj_inputs[1].name: cmd_idx}
                )
            else:
                traj_outs = self.lane_session.run(None, {traj_inputs[0].name: img_input})

            pred_wps   = traj_outs[0][0]                   # (5, 2) normalized [-1, 1]
            wps_pixels = (pred_wps + 1.0) / 2.0 * 224.0   # pixel [0..224]

            # ── 4b. Safety Model Inference (Blocked/Free) ─────────────────────
            prob_blocked_raw = 0.0
            if self.safety_session is not None:
                try:
                    s_inputs = self.safety_session.get_inputs()
                    s_outs   = self.safety_session.run(None, {s_inputs[0].name: img_input})
                    logits   = s_outs[0][0]
                    exp_l    = np.exp(logits - np.max(logits))
                    softmax  = exp_l / np.sum(exp_l)
                    prob_blocked_raw = float(softmax[0])       # class 0 = blocked
                except Exception as e:
                    print(f"[*] Safety model notice: {e}")

            # ── 4c. Smooth the safety probability (EMA) ───────────────────────
            a = self.prob_blocked_ema_alpha
            self._prob_blocked_smoothed = a * prob_blocked_raw + (1.0 - a) * self._prob_blocked_smoothed
            prob_blocked = self._prob_blocked_smoothed

            # ── 5. Pure Pursuit Controller (Trajectory Steering) ──────────────
            Ld_val       = self._eval_param(self.lookahead)
            base_thr_val = self._eval_param(self.throttle)
            steering     = 0.0
            dyn_throttle = base_thr_val
            if self.controller is not None:
                try:
                    steer_res, thr_res, _ = self.controller.update(
                        waypoints=wps_pixels,
                        lookahead_distance=Ld_val,
                        base_throttle=base_thr_val
                    )
                    steering     = float(steer_res)
                    dyn_throttle = float(thr_res)
                except Exception as e:
                    print(f"[*] Controller notice: {e}")

            # ── 6. BLOCKED DETECTION & SAFETY OVERRIDE (with hysteresis) ──────
            now = time.time()
            blk_thresh_enter   = self._eval_param(self.blocked_threshold)
            blk_thresh_release = max(0.0, blk_thresh_enter - self.blocked_release_margin)

            if prob_blocked >= blk_thresh_enter:
                self.blocked_frame_count = min(self.blocked_frame_count + 1, self.confirm_frames + 3)
            else:
                self.blocked_frame_count = max(0, self.blocked_frame_count - 1)

            # Remember the last waypoints seen while confidently FREE and driving
            # normally -- this is the trustworthy reference used to pick evade
            # direction, since the "just blocked" frame is looking at the obstacle.
            if self.evade_state == 'DRIVE' and self.blocked_frame_count == 0:
                self.last_valid_wps = wps_pixels

            is_blocked_active = (
                (prob_blocked >= blk_thresh_enter)
                or (self.blocked_frame_count > 0)
                or (self.evade_state not in ('DRIVE',))
            )

            # Trigger Evade state machine when confirmed (3 frames)
            if self.evade_state == 'DRIVE' and self.blocked_frame_count >= self.confirm_frames:
                reference_wps = self.last_valid_wps if self.last_valid_wps is not None else wps_pixels
                self.evade_steer_dir  = self._compute_evade_direction(active_cmd, reference_wps)
                self.evade_state      = 'REVERSE_TURNING'
                self.evade_start_time = now
                self.evade_attempts   = 1

            # ── 7. EXECUTION PRIORITY DECISION ────────────────────────────────
            if self.evade_state == 'HALTED':
                # Failsafe: exceeded max_evade_attempts. Stay stopped until the
                # operator toggles RUN off/on (which calls reset_evade_fsm()).
                actual_steering = 0.0
                actual_throttle = 0.0

            elif is_blocked_active:
                if self.evade_state == 'DRIVE':
                    # Debounce period before reverse maneuver (frames 1-2 of confirm).
                    # Decay throttle smoothly instead of hard-zeroing it.
                    decay_ratio = self.blocked_frame_count / float(self.confirm_frames)
                    actual_steering = steering * (1.0 - decay_ratio)
                    actual_throttle = dyn_throttle * (1.0 - decay_ratio)

                elif self.evade_state == 'REVERSE_TURNING':
                    if now - self.evade_start_time < self.evade_reverse_duration:
                        actual_steering = self.evade_steer_dir * self.evade_steer_magnitude
                        actual_throttle = -0.22
                    else:
                        self.evade_state      = 'PAUSE'
                        self.evade_start_time = now
                        actual_steering        = 0.0
                        actual_throttle        = 0.0

                elif self.evade_state == 'PAUSE':
                    if now - self.evade_start_time < self.evade_pause_duration:
                        actual_steering = 0.0
                        actual_throttle = 0.0
                    else:
                        self.evade_state      = 'CHECK_FORWARD'
                        self.evade_start_time = now
                        actual_steering        = 0.0
                        actual_throttle        = 0.16

                elif self.evade_state == 'CHECK_FORWARD':
                    if now - self.evade_start_time < self.evade_check_duration:
                        # FIX: blend in live trajectory steering (scaled down) instead of
                        # forcing steering=0.0. This lets the car start correcting back
                        # toward the lane center as soon as it can see the path again,
                        # instead of continuing to point wherever the open-loop reverse
                        # left it aimed -- this is what previously caused the large
                        # drift away from the original route.
                        correction_scale = 0.5
                        actual_steering = steering * correction_scale
                        actual_throttle = 0.16
                    else:
                        if self.blocked_frame_count == 0 and prob_blocked < blk_thresh_release:
                            # Path confirmed clear -> resume normal driving
                            self.evade_state         = 'DRIVE'
                            self.blocked_frame_count = 0
                            self.evade_attempts      = 0
                            actual_steering          = steering
                            actual_throttle          = dyn_throttle
                        elif self.evade_attempts >= self.max_evade_attempts:
                            # Tried enough times -- stop retrying and halt instead of
                            # continuing to drift further off-route.
                            self.evade_state = 'HALTED'
                            actual_steering   = 0.0
                            actual_throttle   = 0.0
                            print(f"\n[!] Evade FAILSAFE: exceeded {self.max_evade_attempts} attempts. "
                                  f"Halting. Toggle RUN off/on to reset.")
                        else:
                            # Still blocked -> repeat evade cycle. Recompute direction
                            # from last_valid_wps again (still the best reference we have).
                            reference_wps = self.last_valid_wps if self.last_valid_wps is not None else wps_pixels
                            self.evade_steer_dir  = self._compute_evade_direction(active_cmd, reference_wps)
                            self.evade_state      = 'REVERSE_TURNING'
                            self.evade_start_time = now
                            self.evade_attempts  += 1
                            actual_steering        = self.evade_steer_dir * self.evade_steer_magnitude
                            actual_throttle        = -0.22
            else:
                # ── 100% FREE PATH: TRAJECTORY MODEL CONTROLS CAR ─────────────
                if is_fsm_stop:
                    actual_steering = 0.0
                    actual_throttle = 0.0
                else:
                    actual_steering = steering
                    actual_throttle = dyn_throttle

            # ── 8. Apply Motion to Hardware ───────────────────────────────────
            final_steering = float(np.clip(actual_steering, -1.0, 1.0))
            final_throttle = float(np.clip(actual_throttle, -0.5, 0.5))

            if self.car is not None:
                self.car.steering = final_steering
                self.car.throttle = final_throttle

            sys.stdout.write(
                f"\r[Urban] RUN | Evade:{self.evade_state:<15} atmpt:{self.evade_attempts} "
                f"| Blk:{prob_blocked*100:4.1f}% (raw:{prob_blocked_raw*100:4.1f}%) "
                f"| S:{final_steering:+.2f} T:{final_throttle:.2f}"
            )
            sys.stdout.flush()

            # ── RELEASE LOCK HERE (Before UI callback) ────────────────────────
            try:
                self._lock.release()
            except RuntimeError:
                pass

            # ── 9. Call UI Callback (Runs OUTSIDE lock) ───────────────────────
            if self.on_frame is not None:
                try:
                    self.on_frame(
                        cv_image=cv_image,
                        waypoints=wps_pixels,
                        prob_blocked=prob_blocked,
                        detections=detections,
                        is_fsm_stop=is_fsm_stop,
                        active_cmd=active_cmd,
                        fsm_status=fsm_status,
                        evade_state=self.evade_state,
                        blocked_count=self.blocked_frame_count,
                        confirm_frames=self.confirm_frames,
                        steering=final_steering,
                        actual_throttle=final_throttle
                    )
                except Exception as e:
                    print(f"\n[!] UI on_frame notice: {e}")
            return  # Lock already released above

        except Exception as e:
            print(f"\n[!] Error in image_callback: {e}")
        finally:
            try:
                self._lock.release()
            except RuntimeError:
                pass