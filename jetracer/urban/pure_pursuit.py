import math
import numpy as np

class PurePursuitController:
    """
    Pure Pursuit Trajectory Tracking Controller for Vehicle Body Frame.
    
    Formula:
        delta = arctan( 2 * L * sin(alpha) / Ld )
        where:
            L = wheelbase length (approx 0.175m for 1/10 RC car)
            Ld = lookahead distance
            alpha = heading angle error to lookahead target point
    """
    def __init__(self, wheelbase=0.175, default_lookahead=0.45):
        self.wheelbase = wheelbase
        self.default_lookahead = default_lookahead
        self.last_steering = 0.0

    def update(self, waypoints, lookahead_distance=None, base_throttle=0.20, max_steering=1.0):
        """
        Inputs:
            waypoints: list or ndarray of N (x, y) coordinates in vehicle body frame
            lookahead_distance: float (Ld)
        Returns:
            steering (float): [-1.0, 1.0]
            dyn_throttle (float): [0.05, 1.0]
            target_point (tuple): (x_target, y_target) chosen lookahead point
        """
        if lookahead_distance is None or lookahead_distance <= 0:
            Ld = self.default_lookahead
        else:
            Ld = lookahead_distance

        if waypoints is None or len(waypoints) == 0:
            return 0.0, base_throttle, (0.0, 0.5)

        waypoints = np.array(waypoints)
        
        # 1. Find lookahead target point closest to distance Ld
        distances = np.sqrt(waypoints[:, 0]**2 + waypoints[:, 1]**2)
        idx = np.argmin(np.abs(distances - Ld))
        target_point = waypoints[idx]
        
        tx, ty = float(target_point[0]), float(target_point[1])
        
        # 2. Pure Pursuit Formula
        # Alpha is the angle between the vehicle heading (Y-axis) and the target point (tx, ty)
        alpha = math.atan2(tx, max(0.01, ty))
        
        # Calculate steering angle delta
        steering_rad = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)
        
        # Map radians to normalized steering range [-1.0, 1.0]
        raw_steering = steering_rad / (math.pi / 4.0) # Assume max steering angle is ~45 deg (pi/4 rad)
        steering = max(-max_steering, min(max_steering, raw_steering))
        
        # Smooth steering transitions
        steering = 0.6 * self.last_steering + 0.4 * steering
        self.last_steering = steering
        
        # 3. Dynamic Adaptive Throttle (reduces speed when turning)
        dyn_throttle = base_throttle - 0.15 * abs(steering)
        dyn_throttle = max(0.05, min(1.0, dyn_throttle))
        
        return float(steering), float(dyn_throttle), (tx, ty)
