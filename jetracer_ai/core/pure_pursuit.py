"""
Pure Pursuit Trajectory Steering Controller for JetRacer.
"""
import math
import numpy as np


class PurePursuitController:
    """
    Balanced & Precise Trajectory Steering Controller for JetRacer.
    """
    def __init__(
        self,
        wheelbase=0.14,
        default_lookahead=0.40,
        gain=2.2,
        heading_weight=0.6,
        deadband=0.12,
        smoothing=0.5,
        **kwargs
    ):
        if 'wheel_base' in kwargs:
            wheelbase = kwargs['wheel_base']
        if 'lookahead_distance' in kwargs:
            default_lookahead = kwargs['lookahead_distance']

        self.wheelbase = wheelbase
        self.default_lookahead = default_lookahead
        self.gain = gain
        self.heading_weight = heading_weight
        self.deadband = deadband
        self.smoothing = smoothing
        self.last_steering = 0.0

    def update(self, waypoints, lookahead_distance=None, base_throttle=0.20, max_steering=1.0):
        if waypoints is None or len(waypoints) == 0:
            return 0.0, base_throttle, (112.0, 112.0)

        waypoints = np.array(waypoints, dtype=np.float32)
        n = len(waypoints)

        if n == 5:
            weights = np.array([5.0, 3.5, 2.0, 1.0, 0.5], dtype=np.float32)
        else:
            weights = np.arange(n, 0, -1, dtype=np.float32) ** 1.3

        weight_sum = np.sum(weights)
        lateral_errors = (waypoints[:, 0] - 112.0) / 112.0
        weighted_lat_err = float(np.sum(weights * lateral_errors) / weight_sum)

        abs_lat = abs(weighted_lat_err)
        sign_lat = 1.0 if weighted_lat_err >= 0 else -1.0
        scaled_lat_err = sign_lat * (abs_lat ** 0.85)

        mid_idx = min(2, n - 1)
        near_pt = waypoints[0]
        mid_pt = waypoints[mid_idx]

        dx = float(mid_pt[0] - near_pt[0])
        dy = float(near_pt[1] - mid_pt[1])

        heading_angle_rad = math.atan2(dx, max(1.0, dy))
        heading_steer = heading_angle_rad / (math.pi / 4.0)

        raw_steering = self.gain * scaled_lat_err + self.heading_weight * heading_steer

        if abs(raw_steering) > 0.015:
            sign_s = 1.0 if raw_steering > 0 else -1.0
            abs_s = min(1.0, abs(raw_steering))
            boosted_s = sign_s * (self.deadband + (1.0 - self.deadband) * abs_s)
        else:
            boosted_s = 0.0

        boosted_s = max(-max_steering, min(max_steering, boosted_s))
        steering = (1.0 - self.smoothing) * self.last_steering + self.smoothing * boosted_s
        self.last_steering = steering
        steering = float(np.clip(steering, -max_steering, max_steering))

        dyn_throttle = base_throttle - 0.12 * abs(steering)
        dyn_throttle = float(max(0.05, min(1.0, dyn_throttle)))

        target_point = (float(waypoints[-1][0]), float(waypoints[-1][1]))

        return steering, dyn_throttle, target_point

    def compute_steering(self, waypoints, lookahead_distance=None, base_throttle=0.20, max_steering=1.0):
        steering, dyn_throttle, target_point = self.update(waypoints, lookahead_distance, base_throttle, max_steering)
        return steering, target_point
