#!/usr/bin/env python3
"""
RacerController - Lp tru tng iu khin JetRacer (Ackermann Steering)

Xe JetRacer dng Ackermann steering (li servo pha trc + motor ga pha sau),
KHÔNG PHẢI differential drive nh JetBot.

Khc bit chnh:
  - JetBot:    robot.set_motors(left_speed, right_speed)  → quay ti ch c
  - JetRacer:  car.steering = angle, car.throttle = speed → KHÔNG quay ti ch

Lp ny cung cp API thng nht  code chnh khng cn quan tm loi xe.

Cch dng:
    from src.core.control.racer_controller import RacerController
    controller = RacerController()
    controller.forward(0.3)          # i thng
    controller.steer(0.5, 0.3)       # R phi nh + i ti
    controller.turn_angle(90)        # R phi 90  (i vng cung)
    controller.stop()                # Dng
"""

import sys
# (translated)
sys.path = [p for p in sys.path if 'python2.7' not in p]

import time
import math

try:
    import rospy
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# ============================================================
# (translated)
# ============================================================
DEFAULT_CONFIG = {
    # --- Throttle (ga) ---
    "BASE_THROTTLE": 0.20,         # Tc  i thng mc nh (0.0 → 1.0)
    "TURN_THROTTLE": 0.15,         # Tc  khi ang r (chm hn)
    "MAX_THROTTLE": 0.40,          # Gii hn tc  ti a (an ton)

    # (translated)
    "STEERING_GAIN": 0.8,          # H s khuch i gc li khi bm line
    "MAX_STEERING": 1.0,           # Gii hn gc li ti a (-1.0 tri ↔ +1.0 phi)
    "STEERING_OFFSET": 0.0,        # B lch nu xe b lch (calibrate trn xe tht)

    # (translated)
    "TURN_DURATION_90_DEG": 1.5,   # Thi gian (giy)  r 90°  TURN_THROTTLE
    "STEERING_VALUE_FOR_TURN": 0.7, # Gi tr steering khi r gp (0.0→1.0)

    # --- PID Controller ---
    "PID_KP": 0.5,                 # Proportional gain
    "PID_KI": 0.0,                 # Integral gain (thng  0 cho robot nh)
    "PID_KD": 0.1,                 # Derivative gain (gim dao ng)

    # --- Safety ---
    "SAFE_ZONE_PERCENT": 0.3,      # Vng an ton  gia (% chiu rng nh)
}


class RacerController:
    """
    Controller cho JetRacer (Waveshare JetRacer Pro AI Kit)
    
    Hardware:
    - Servo li (steering): PCA9685 channel, range -1.0 (tri) → +1.0 (phi)
    - Motor ga (throttle):  PCA9685 channel, range -1.0 (li) → +1.0 (tin)
    - RPLIDAR trn nc
    - CSI Camera pha trc
    """

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.car = None
        self._mock = False

        # PID state
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = time.time()

        self._initialize_hardware()

    def _initialize_hardware(self):
        """(see module docstring)"""
        # (translated)
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self.car = NvidiaRacecar()
            self.car.steering = 0.0
            self.car.throttle = 0.0
            self._log("Khi to JetRacer (NvidiaRacecar) thnh cng.")
            return
        except Exception as e:
            self._log(f"Khng tm thy jetracer library: {e}", level="warn")

        # (translated)
        try:
            from jetbot import Robot
            self.car = Robot()
            self._mock = False
            self._log("Khi to JetBot Pro (fallback) thnh cng.")
            self._log("LƯU Ý: ang dng JetBot API trn JetRacer - cn kim tra tng thch!", level="warn")
            return
        except Exception as e:
            self._log(f"Khng tm thy jetbot library: {e}", level="warn")

        # (translated)
        self._log("Khng tm thy phn cng → Chy  ch  MÔ PHỎNG (Mock).", level="warn")
        from unittest.mock import Mock
        self.car = Mock()
        self._mock = True

    # ============================================================
    # (translated)
    # ============================================================

    def forward(self, speed=None):
        """(see module docstring)"""
        speed = speed or self.config["BASE_THROTTLE"]
        speed = self._clamp_throttle(speed)
        self._set_steering(0.0)
        self._set_throttle(speed)

    def stop(self):
        """(see module docstring)"""
        self._set_throttle(0.0)
        self._set_steering(0.0)

    def steer(self, steering_value, speed=None):
        """
        i vi gc li cho trc.
        
        Args:
            steering_value: -1.0 (tri max) → 0.0 (thng) → +1.0 (phi max)
            speed: tc , mc nh BASE_THROTTLE
        """
        speed = speed or self.config["BASE_THROTTLE"]
        speed = self._clamp_throttle(speed)
        steering_value = self._clamp_steering(steering_value)
        self._set_steering(steering_value)
        self._set_throttle(speed)

    # ============================================================
    # (translated)
    # ============================================================

    def turn_angle(self, degrees, record_callback=None):
        """
        R mt gc cho trc (i vng cung, KHÔNG quay ti ch).
        
        JetRacer khng th quay ti ch nh JetBot, nn phi:
        1. nh li sang mt bn
        2. i ti vi tc  chm
        3. i  thi gian
        4. Tr li thng + dng

        Args:
            degrees: gc r (dng = phi, m = tri)
            record_callback: hm ghi video debug (gi mi frame)
        """
        if degrees == 0:
            return

        # (translated)
        duration = abs(degrees) / 90.0 * self.config["TURN_DURATION_90_DEG"]
        turn_steering = self.config["STEERING_VALUE_FOR_TURN"]
        turn_throttle = self.config["TURN_THROTTLE"]

        # (translated)
        if degrees > 0:
            self._set_steering(turn_steering)      # Li phi
        else:
            self._set_steering(-turn_steering)     # Li tri

        # (translated)
        self._set_throttle(turn_throttle)

        # (translated)
        start_time = time.time()
        while time.time() - start_time < duration:
            if record_callback:
                record_callback()
            time.sleep(0.05)  # 20 FPS

        # (translated)
        self.stop()
        time.sleep(0.3)

        if record_callback:
            record_callback()

    def correct_course_pid(self, error, image_width):
        """
        PID Controller cho bm line.
        
        Thay th P-controller c bng PID y .
        u ra l gi tr steering (-1.0 → +1.0) thay v chnh lch tc  motor.
        
        Args:
            error: sai lch pixel t tm nh (dng = lch phi)
            image_width: chiu rng nh (pixels)
        """
        # (translated)
        normalized_error = error / (image_width / 2.0)

        # (translated)
        if abs(normalized_error) < self.config["SAFE_ZONE_PERCENT"]:
            self.forward()
            return

        # PID calculation
        current_time = time.time()
        dt = current_time - self._pid_last_time
        if dt <= 0:
            dt = 0.05  # fallback

        # P - Proportional
        p_term = self.config["PID_KP"] * normalized_error

        # (translated)
        self._pid_integral += normalized_error * dt
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral))  # clamp
        i_term = self.config["PID_KI"] * self._pid_integral

        # D - Derivative
        d_error = (normalized_error - self._pid_last_error) / dt
        d_term = self.config["PID_KD"] * d_error

        # (translated)
        steering_output = p_term + i_term + d_term
        steering_output = self._clamp_steering(steering_output)

        # (translated)
        self._pid_last_error = normalized_error
        self._pid_last_time = current_time

        # (translated)
        self.steer(steering_output, self.config["BASE_THROTTLE"])

    def reset_pid(self):
        """(see module docstring)"""
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = time.time()

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _set_throttle(self, value):
        """(see module docstring)"""
        value = self._clamp_throttle(value)
        if hasattr(self.car, 'throttle'):
            # NvidiaRacecar API
            self.car.throttle = value
        elif hasattr(self.car, 'set_motors'):
            # (translated)
            self.car.set_motors(value, value)
        elif hasattr(self.car, 'forward'):
            if value > 0:
                self.car.forward(value)
            elif value < 0:
                self.car.backward(abs(value))
            else:
                self.car.stop()

        if self._mock and HAS_ROS:
            self.current_throttle = value
            if not hasattr(self, 'current_steering'):
                self.current_steering = 0.0
            if hasattr(rospy, 'publish_control'):
                try:
                    rospy.publish_control(self.current_steering, self.current_throttle)
                except Exception:
                    pass

    def _set_steering(self, value):
        """(see module docstring)"""
        value = self._clamp_steering(value)
        value += self.config["STEERING_OFFSET"]
        if hasattr(self.car, 'steering'):
            # NvidiaRacecar API
            self.car.steering = value
        elif hasattr(self.car, 'set_motors') and not self._mock:
            # (translated)
            # (translated)
            # (translated)
            pass  # Steering s c x l trong _set_throttle

        if self._mock and HAS_ROS:
            self.current_steering = value
            if not hasattr(self, 'current_throttle'):
                self.current_throttle = 0.0
            if hasattr(rospy, 'publish_control'):
                try:
                    rospy.publish_control(self.current_steering, self.current_throttle)
                except Exception:
                    pass

    def _clamp_throttle(self, value):
        return max(-self.config["MAX_THROTTLE"], min(self.config["MAX_THROTTLE"], value))

    def _clamp_steering(self, value):
        return max(-self.config["MAX_STEERING"], min(self.config["MAX_STEERING"], value))

    def _log(self, msg, level="info"):
        if HAS_ROS:
            if level == "warn":
                rospy.logwarn(msg)
            elif level == "error":
                rospy.logerr(msg)
            else:
                rospy.loginfo(msg)
        else:
            print(f"[{level.upper()}] {msg}")


# ============================================================
# (translated)
# ============================================================
if __name__ == "__main__":
    print("=== Test RacerController ===")
    ctrl = RacerController()
    print("Forward...")
    ctrl.forward(0.2)
    time.sleep(1)
    print("Steer right...")
    ctrl.steer(0.5, 0.2)
    time.sleep(1)
    print("Turn 90 degrees...")
    ctrl.turn_angle(90)
    print("Stop.")
    ctrl.stop()
    print("=== Test complete ===")
