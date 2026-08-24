"""
Stanley & PID Steering Controllers for Lane Following.
"""
import time
import math
from jetracer_ai.core.kalman import KalmanFilter1D


class PIDController:
    """PID Steering Controller with integrated 1D Kalman Filter."""
    def __init__(self, q_process=0.01, r_measure=0.1):
        self.kf = KalmanFilter1D(q_process=q_process, r_measure=r_measure)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_time = time.time()
        self.kf.reset()

    @property
    def smoothed_x(self):
        return self.kf.x

    def update(self, raw_x, kp, ki, kd, alpha=0.7, bias=0.0, r_measure=None, q_process=None):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05
        self.last_time = now

        if r_measure is not None:
            self.kf.r_measure = r_measure
        if q_process is not None:
            self.kf.q_process = q_process
        elif alpha is not None and alpha > 0:
            self.kf.r_measure = max(0.001, (1.0 - alpha) * 0.2)

        filtered_x, estimated_v = self.kf.update(raw_x, dt)
        error = filtered_x

        self.integral += error * dt
        max_int = 0.3 / (ki + 1e-9)
        self.integral = max(-max_int, min(max_int, self.integral))

        derivative = estimated_v
        steering = kp * error + ki * self.integral + kd * derivative + bias
        return max(-1.0, min(1.0, steering))


class StanleyController:
    """
    Stanley Steering Controller using Kalman-filtered Cross-Track Error (CTE)
    and Kalman estimated lateral velocity for heading error estimation.
    """
    def __init__(self, q_process=0.01, r_measure=0.1):
        self.kf = KalmanFilter1D(q_process=q_process, r_measure=r_measure)
        self.reset()

    def reset(self):
        self.last_time = time.time()
        self.kf.reset()

    @property
    def smoothed_x(self):
        return self.kf.x

    def update(self, raw_x, k=1.2, base_throttle=0.20, brake_gain=0.10, bias=0.0, alpha=0.7, lidar_offset=0.0):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0.0 or dt > 0.5:
            dt = 0.05
        self.last_time = now

        if alpha is not None and alpha > 0:
            self.kf.r_measure = max(0.001, (1.0 - alpha) * 0.2)

        effective_x = raw_x + (lidar_offset if lidar_offset is not None else 0.0)
        effective_x = max(-1.0, min(1.0, effective_x))

        cte, vx = self.kf.update(effective_x, dt)
        heading_error = math.atan2(vx, 1.0)

        speed = max(0.05, base_throttle)
        cte_correction = math.atan2(k * cte, speed + 0.05)

        steering = heading_error + cte_correction + bias
        steering = max(-1.0, min(1.0, steering))

        dyn_throttle = base_throttle - brake_gain * abs(steering)
        dyn_throttle = max(0.0, min(1.0, dyn_throttle))

        return steering, dyn_throttle


class StanleyController1D(StanleyController):
    """Alias for StanleyController."""
    pass
