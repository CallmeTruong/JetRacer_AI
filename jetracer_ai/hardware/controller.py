import time
import atexit
import logging

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except ImportError:
    try:
        from jetracer_ai.hardware.racecar import NvidiaRacecar
    except ImportError:
        NvidiaRacecar = None

logger = logging.getLogger("RacecarController")


class RacecarController:
    """
    Hardware Abstraction Layer (Motor Controller)
    Manages hardware operations for JetRacer on Jetson Nano.
    Preserves exact control logic and configuration parameters from jetracer-car.
    """

    def __init__(
        self,
        base_throttle=0.15,     # Base throttle value for road following (0.0 to 1.0)
        cm_per_second=30.0,     # Estimated vehicle speed calibration: 30cm per second
        straight_steering=0.0,  # Straight steering angle
        left_steering=-1.0,     # Left turn steering angle
        right_steering=1.0,     # Right turn steering angle
        max_throttle=0.5,       # Absolute safety limit for throttle regardless of speed factor
        turn90_forward_duration_left=3.0,
        turn90_reverse_duration_left=3.0,
        turn90_forward_duration_right=3.0,
        turn90_reverse_duration_right=3.0,
        pause_between_phases=0.15,
        kick_throttle=0.5,
        kick_duration=0.2,
    ):
        if NvidiaRacecar is not None:
            self.car = NvidiaRacecar()
        else:
            self.car = None
            logger.warning("NvidiaRacecar library not found. Running in mock/simulation mode.")

        self.base_throttle = self._clamp(base_throttle, -1.0, 1.0)
        self.cm_per_second = max(float(cm_per_second), 1e-6)
        self.max_throttle = self._clamp(max_throttle, 0.0, 1.0)

        self.steering_config = {
            "STRAIGHT": self._clamp(straight_steering, -1.0, 1.0),
            "LEFT": self._clamp(left_steering, -1.0, 1.0),
            "RIGHT": self._clamp(right_steering, -1.0, 1.0),
        }

        self.turn90_config = {
            "LEFT": {
                "forward": max(0.0, float(turn90_forward_duration_left)),
                "reverse": max(0.0, float(turn90_reverse_duration_left)),
            },
            "RIGHT": {
                "forward": max(0.0, float(turn90_forward_duration_right)),
                "reverse": max(0.0, float(turn90_reverse_duration_right)),
            },
        }
        self.pause_between_phases = max(0.0, float(pause_between_phases))
        self.kick_throttle = self._clamp(kick_throttle, 0.0, 1.0)
        self.kick_duration = max(0.0, float(kick_duration))

        self.stop()
        atexit.register(self.stop)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, float(value)))

    def _safe_throttle(self, value):
        return self._clamp(value, -self.max_throttle, self.max_throttle)

    def set_steering(self, value):
        if self.car is not None:
            self.car.steering = self._clamp(value, -1.0, 1.0)

    def set_throttle(self, value):
        if self.car is not None:
            self.car.throttle = self._safe_throttle(value)

    def stop(self):
        if self.car is not None:
            self.car.throttle = 0.0
            self.car.steering = self.steering_config["STRAIGHT"]

    def _run_timed(self, steering, throttle, duration, kick_throttle=0.0, kick_duration=0.0):
        target_throttle = self._safe_throttle(throttle)
        kick_t = self._safe_throttle(kick_throttle) if kick_throttle != 0.0 else 0.0

        start_time = time.time()
        effective_kick_dur = min(max(0.0, float(kick_duration)), max(0.0, float(duration)))

        self.set_steering(steering)

        if kick_t != 0.0 and effective_kick_dur > 0.0:
            self.set_throttle(kick_t)
            time.sleep(effective_kick_dur)

        remaining = duration - effective_kick_dur
        if remaining > 0:
            self.set_throttle(target_throttle)
            time.sleep(remaining)

    def move_distance_cm(self, distance_cm, speed_factor=1.0):
        speed_factor = self._clamp(speed_factor, -2.0, 2.0)
        dist = float(distance_cm)

        duration = abs(dist) / (self.cm_per_second * abs(speed_factor)) if speed_factor != 0 else 0
        direction = 1.0 if dist >= 0 else -1.0
        throttle = self.base_throttle * speed_factor * direction

        kick_t = self.kick_throttle * direction if self.kick_throttle > 0 else 0.0

        try:
            self._run_timed(
                steering=self.steering_config["STRAIGHT"],
                throttle=throttle,
                duration=duration,
                kick_throttle=kick_t,
                kick_duration=self.kick_duration,
            )
        finally:
            self.stop()

    # Alias compatible with decision.py
    def move_cm(self, distance_cm, speed_factor=1.0):
        self.move_distance_cm(distance_cm, speed_factor)

    def _turn_90_2phase(self, direction):
        direction = direction.upper()
        if direction not in ["LEFT", "RIGHT"]:
            raise ValueError("direction must be 'LEFT' or 'RIGHT'")

        steer_fwd = self.steering_config[direction]
        steer_rev = self.steering_config["RIGHT" if direction == "LEFT" else "LEFT"]

        dur_fwd = self.turn90_config[direction]["forward"]
        dur_rev = self.turn90_config[direction]["reverse"]

        try:
            # Phase 1: Forward motion with same-direction steering
            if dur_fwd > 0:
                self._run_timed(
                    steering=steer_fwd,
                    throttle=self.base_throttle,
                    duration=dur_fwd,
                    kick_throttle=self.kick_throttle,
                    kick_duration=self.kick_duration,
                )

            if self.pause_between_phases > 0:
                self.stop()
                time.sleep(self.pause_between_phases)

            # Phase 2: Reverse motion with counter-steering
            if dur_rev > 0:
                self._run_timed(
                    steering=steer_rev,
                    throttle=-self.base_throttle,
                    duration=dur_rev,
                    kick_throttle=-self.kick_throttle,
                    kick_duration=self.kick_duration,
                )
        finally:
            self.stop()

    def turn_90_left(self):
        self._turn_90_2phase("LEFT")

    def turn_90_right(self):
        self._turn_90_2phase("RIGHT")

    def turn_left(self, duration=1.2):
        try:
            self._run_timed(
                steering=self.steering_config["LEFT"],
                throttle=self.base_throttle,
                duration=duration,
                kick_throttle=self.kick_throttle,
                kick_duration=self.kick_duration,
            )
        finally:
            self.stop()

    def turn_right(self, duration=1.0):
        try:
            self._run_timed(
                steering=self.steering_config["RIGHT"],
                throttle=self.base_throttle,
                duration=duration,
                kick_throttle=self.kick_throttle,
                kick_duration=self.kick_duration,
            )
        finally:
            self.stop()
