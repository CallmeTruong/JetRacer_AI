#!/usr/bin/env python3
"""
RacerController - Abstraction layer for JetRacer (Ackermann Steering)

The JetRacer uses Ackermann steering (front servo steering + rear motor drive),
NOT differential drive like JetBot.

Key differences:
  - JetBot:    robot.set_motors(left_speed, right_speed)  -> can spin in place
  - JetRacer:  car.steering = angle, car.throttle = speed -> CANNOT spin in place

This class provides a unified API so the main code doesn't need to worry about vehicle type.

Usage:
    from src.core.control.racer_controller import RacerController
    controller = RacerController()
    controller.forward(0.3)          # Drive straight
    controller.steer(0.5, 0.3)       # Slight right turn + drive forward
    controller.turn_angle(90)        # Turn right 90 degrees (arc turn)
    controller.stop()                # Stop
"""

import sys
# Remove Python 2.7 paths to avoid import conflicts
sys.path = [p for p in sys.path if 'python2.7' not in p]

import time
import math

try:
    import rospy
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# ============================================================
# Default Configuration Parameters
# ============================================================
DEFAULT_CONFIG = {
    # --- Throttle ---
    "BASE_THROTTLE": 0.20,         # Default straight-line speed (0.0 -> 1.0)
    "TURN_THROTTLE": 0.15,         # Speed while turning (slower)
    "MAX_THROTTLE": 0.40,          # Maximum throttle limit (safety)

    # --- Steering ---
    "STEERING_GAIN": 0.8,          # Steering gain when following lane
    "MAX_STEERING": 1.0,           # Maximum steering angle (-1.0 left <-> +1.0 right)
    "STEERING_OFFSET": 0.0,        # Steering offset if vehicle drifts (calibrate on real car)

    # --- Arc Turning ---
    "TURN_DURATION_90_DEG": 1.5,   # Time (seconds) to complete a 90-degree arc turn at TURN_THROTTLE
    "STEERING_VALUE_FOR_TURN": 0.7, # Steering value for sharp turns (0.0 -> 1.0)

    # --- PID Controller ---
    "PID_KP": 0.5,                 # Proportional gain
    "PID_KI": 0.0,                 # Integral gain (typically 0 for small robots)
    "PID_KD": 0.1,                 # Derivative gain (reduce oscillations)

    # --- Safety ---
    "SAFE_ZONE_PERCENT": 0.3,      # Center safe zone (% of image width)
}


class RacerController:
    """
    Controller for JetRacer (Waveshare JetRacer Pro AI Kit)
    
    Hardware:
    - Servo steering: PCA9685 channel, range -1.0 (left) -> +1.0 (right)
    - Motor throttle: PCA9685 channel, range -1.0 (reverse) -> +1.0 (forward)
    - RPLIDAR on top
    - CSI Camera at front
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
        """Try to initialize the JetRacer hardware, falling back to JetBot or Mock."""
        # Attempt 1: NvidiaRacecar (JetRacer)
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self.car = NvidiaRacecar()
            self.car.steering = 0.0
            self.car.throttle = 0.0
            self._log("Initialized JetRacer (NvidiaRacecar) successfully.")
            return
        except Exception as e:
            self._log(f"jetracer library not found: {e}", level="warn")

        # Attempt 2: JetBot (fallback)
        try:
            from jetbot import Robot
            self.car = Robot()
            self._mock = False
            self._log("Initialized JetBot Pro (fallback) successfully.")
            self._log("WARNING: Using JetBot API on JetRacer - check compatibility!", level="warn")
            return
        except Exception as e:
            self._log(f"jetbot library not found: {e}", level="warn")

        # Attempt 3: Mock (simulation)
        self._log("No hardware found -> Running in SIMULATION (Mock) mode.", level="warn")
        from unittest.mock import Mock
        self.car = Mock()
        self._mock = True

    # ============================================================
    # Basic Movement API
    # ============================================================

    def forward(self, speed=None):
        """Drive straight forward at the given speed."""
        speed = speed or self.config["BASE_THROTTLE"]
        speed = self._clamp_throttle(speed)
        self._set_steering(0.0)
        self._set_throttle(speed)

    def stop(self):
        """Stop the vehicle immediately (zero throttle and center steering)."""
        self._set_throttle(0.0)
        self._set_steering(0.0)

    def steer(self, steering_value, speed=None):
        """
        Drive with a given steering angle.
        
        Args:
            steering_value: -1.0 (full left) -> 0.0 (straight) -> +1.0 (full right)
            speed: throttle speed, defaults to BASE_THROTTLE
        """
        speed = speed or self.config["BASE_THROTTLE"]
        speed = self._clamp_throttle(speed)
        steering_value = self._clamp_steering(steering_value)
        self._set_steering(steering_value)
        self._set_throttle(speed)

    # ============================================================
    # Arc Turning (Ackermann-style, cannot spin in place)
    # ============================================================

    def turn_angle(self, degrees, record_callback=None):
        """
        Execute an arc turn of the given angle (cannot spin in place).
        
        Since JetRacer cannot spin in place like JetBot, the procedure is:
        1. Set steering to one side
        2. Drive forward at low speed
        3. Wait for the calculated duration
        4. Straighten steering and stop

        Args:
            degrees: turn angle (positive = right, negative = left)
            record_callback: optional function called each frame for debug recording
        """
        if degrees == 0:
            return

        # Calculate duration proportional to angle
        duration = abs(degrees) / 90.0 * self.config["TURN_DURATION_90_DEG"]
        turn_steering = self.config["STEERING_VALUE_FOR_TURN"]
        turn_throttle = self.config["TURN_THROTTLE"]

        # Set steering direction
        if degrees > 0:
            self._set_steering(turn_steering)      # Steer right
        else:
            self._set_steering(-turn_steering)     # Steer left

        # Apply throttle
        self._set_throttle(turn_throttle)

        # Wait for turn to complete
        start_time = time.time()
        while time.time() - start_time < duration:
            if record_callback:
                record_callback()
            time.sleep(0.05)  # 20 FPS

        # Stop and pause briefly
        self.stop()
        time.sleep(0.3)

        if record_callback:
            record_callback()

    def correct_course_pid(self, error, image_width):
        """
        PID Controller for lane following.
        
        Replaces the old P-controller with a full PID implementation.
        Output is a steering value (-1.0 -> +1.0) instead of motor speed difference.
        
        Args:
            error: pixel offset from image center (positive = offset right)
            image_width: image width (pixels)
        """
        # Normalize error to [-1.0, 1.0]
        normalized_error = error / (image_width / 2.0)

        # If within safe zone, drive straight
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

        # I - Integral (with anti-windup clamping)
        self._pid_integral += normalized_error * dt
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral))  # clamp
        i_term = self.config["PID_KI"] * self._pid_integral

        # D - Derivative
        d_error = (normalized_error - self._pid_last_error) / dt
        d_term = self.config["PID_KD"] * d_error

        # Combine PID terms
        steering_output = p_term + i_term + d_term
        steering_output = self._clamp_steering(steering_output)

        # Update PID state
        self._pid_last_error = normalized_error
        self._pid_last_time = current_time

        # Apply steering command
        self.steer(steering_output, self.config["BASE_THROTTLE"])

    def reset_pid(self):
        """Reset PID controller internal state."""
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = time.time()

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _set_throttle(self, value):
        """Set throttle via the appropriate hardware API."""
        value = self._clamp_throttle(value)
        if hasattr(self.car, 'throttle'):
            # NvidiaRacecar API
            self.car.throttle = value
        elif hasattr(self.car, 'set_motors'):
            # JetBot fallback (differential drive)
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
        """Set steering angle via the appropriate hardware API."""
        value = self._clamp_steering(value)
        value += self.config["STEERING_OFFSET"]
        if hasattr(self.car, 'steering'):
            # NvidiaRacecar API
            self.car.steering = value
        elif hasattr(self.car, 'set_motors') and not self._mock:
            # JetBot does not have separate steering; handled in _set_throttle
            pass

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
# Standalone Test
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
