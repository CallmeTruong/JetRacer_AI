import math
import numpy as np


class PurePursuitController:
    """
    Balanced & Precise Trajectory Steering Controller for JetRacer.

    FIXES vs original:
      1. Lateral-error weights now favor the NEAR waypoints (most reliable),
         instead of the far waypoint (least reliable, causes early/aggressive turns).
      2. Heading angle is computed between a near-mid pair of waypoints instead of
         waypoint[0] -> waypoint[-1], so the far point's noise isn't counted twice
         (once in lateral error, once in heading).
      3. Deadband lowered by default (0.15 -> 0.12) and gain lowered (3.5 -> 2.2 typical)
         so small lateral errors don't get boosted into large steering angles.
      4. Steering low-pass filter smoothing increased (0.35/0.65 -> 0.5/0.5) to reduce
         jerky corrections frame-to-frame.
    """
    def __init__(
        self,
        wheelbase=0.14,
        default_lookahead=0.40,
        gain=2.2,               # Reduced from 3.5: avoids over-aggressive turning
        heading_weight=0.6,      # Reduced from 1.0: heading no longer double-counts far point
        deadband=0.12,           # Reduced from 0.25: avoids over-boosting small errors
        smoothing=0.5,           # New: fraction of *new* steering value used each frame
        **kwargs
    ):
        if 'wheel_base' in kwargs:
            wheelbase = kwargs['wheel_base']
        if 'lookahead_distance' in kwargs:
            default_lookahead = kwargs['lookahead_distance']

        self.wheelbase         = wheelbase
        self.default_lookahead = default_lookahead
        self.gain              = gain
        self.heading_weight    = heading_weight
        self.deadband          = deadband
        self.smoothing         = smoothing
        self.last_steering     = 0.0

    def update(self, waypoints, lookahead_distance=None, base_throttle=0.20, max_steering=1.0):
        """
        Inputs:
            waypoints: ndarray (N, 2) in PIXEL space [0..224]
                       x=horizontal [0..224], y=vertical [0..224] (0=top, 224=bottom)
            base_throttle: float base throttle speed
            max_steering: float max steering clamp

        Returns:
            steering (float): [-1.0, 1.0]  (negative = LEFT, positive = RIGHT)
            dyn_throttle (float): [0.05, 1.0]
            target_point (tuple): (tx_pixel, ty_pixel) for UI display
        """
        if waypoints is None or len(waypoints) == 0:
            return 0.0, base_throttle, (112.0, 112.0)

        waypoints = np.array(waypoints, dtype=np.float32)
        n = len(waypoints)

        # ── 1. NEAR-Weighted Lateral Error ─────────────────────────────────────
        # Waypoint 0 = nearest to car (most trustworthy, least perspective distortion).
        # Waypoint -1 = farthest (least trustworthy). Weight the NEAR points more so
        # the car reacts to what's actually in front of it, not a noisy far guess.
        if n == 5:
            weights = np.array([5.0, 3.5, 2.0, 1.0, 0.5], dtype=np.float32)
        else:
            # Descending weights: near points matter most
            weights = np.arange(n, 0, -1, dtype=np.float32) ** 1.3

        weight_sum = np.sum(weights)

        # Normalized lateral errors from center (112.0): -1.0 = left, +1.0 = right
        lateral_errors = (waypoints[:, 0] - 112.0) / 112.0
        weighted_lat_err = float(np.sum(weights * lateral_errors) / weight_sum)

        # Progressive Power Scaling (|err|^0.85): smooth curve escalation
        abs_lat  = abs(weighted_lat_err)
        sign_lat = 1.0 if weighted_lat_err >= 0 else -1.0
        scaled_lat_err = sign_lat * (abs_lat ** 0.85)

        # ── 2. Trajectory Heading Slope Angle (near-mid segment only) ─────────
        # Use a NEAR-MID pair instead of near->far, so we don't double count the
        # far (noisiest) waypoint that's already dominant risk in lateral error.
        mid_idx = min(2, n - 1)   # waypoint index ~middle of the trajectory
        near_pt = waypoints[0]
        mid_pt  = waypoints[mid_idx]

        dx = float(mid_pt[0] - near_pt[0])
        dy = float(near_pt[1] - mid_pt[1])  # Y is 0 at top, 224 at bottom

        heading_angle_rad = math.atan2(dx, max(1.0, dy))
        heading_steer      = heading_angle_rad / (math.pi / 4.0)

        # ── 3. Combined Raw Steering Angle ────────────────────────────────────
        raw_steering = self.gain * scaled_lat_err + self.heading_weight * heading_steer

        # ── 4. Servo Deadband Compensation ────────────────────────────────────
        if abs(raw_steering) > 0.015:
            sign_s    = 1.0 if raw_steering > 0 else -1.0
            abs_s     = min(1.0, abs(raw_steering))
            boosted_s = sign_s * (self.deadband + (1.0 - self.deadband) * abs_s)
        else:
            boosted_s = 0.0

        # Clip steering to max limits [-1.0, 1.0]
        boosted_s = max(-max_steering, min(max_steering, boosted_s))

        # ── 5. Smooth Exponential Low-Pass Filter ─────────────────────────────
        steering = (1.0 - self.smoothing) * self.last_steering + self.smoothing * boosted_s
        self.last_steering = steering
        steering = float(np.clip(steering, -max_steering, max_steering))

        # ── 6. Adaptive Throttle Reduction on Turn ────────────────────────────
        dyn_throttle = base_throttle - 0.12 * abs(steering)
        dyn_throttle = float(max(0.05, min(1.0, dyn_throttle)))

        target_point = (float(waypoints[-1][0]), float(waypoints[-1][1]))

        return steering, dyn_throttle, target_point

    def compute_steering(self, waypoints, lookahead_distance=None, base_throttle=0.20, max_steering=1.0):
        steering, dyn_throttle, target_point = self.update(waypoints, lookahead_distance, base_throttle, max_steering)
        return steering, target_point