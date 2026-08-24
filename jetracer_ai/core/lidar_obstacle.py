import math
import numpy as np
import threading

class LiDARObstacleAvoidance:
    """
    ROS LiDAR LaserScan Obstacle Avoidance Processor for JetRacer.
    Subscribes to ROS /scan topic (sensor_msgs/LaserScan).
    
    Calculates continuous evasive steering offset:
        - Scans front 120-degree arc (-60 deg to +60 deg)
        - If obstacle is detected closer than alert_distance (e.g. 0.70m):
            - If obstacle is on Left  -> returns positive offset (+0.35 max) -> Steers Right
            - If obstacle is on Right -> returns negative offset (-0.35 max) -> Steers Left
        - If clear (> 0.70m) -> returns 0.0 offset
    """
    def __init__(self, alert_distance=0.70, max_offset=0.35, scan_arc_deg=60.0):
        self.alert_distance = alert_distance
        self.max_offset = max_offset
        self.scan_arc_rad = math.radians(scan_arc_deg)
        
        self.current_offset = 0.0
        self.min_left_dist = 999.0
        self.min_right_dist = 999.0
        self.obstacle_detected = False
        
        self._lock = threading.Lock()

    def laser_callback(self, msg):
        """
        ROS /scan topic callback for sensor_msgs/LaserScan.
        """
        if not self._lock.acquire(blocking=False):
            return

        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            angle_min = msg.angle_min
            angle_increment = msg.angle_increment
            
            # Filter valid finite ranges
            num_readings = len(ranges)
            angles = angle_min + np.arange(num_readings) * angle_increment
            
            # Normalize angles to [-pi, pi]
            angles = (angles + math.pi) % (2 * math.pi) - math.pi
            
            # Filter front left arc (0 to +scan_arc_rad) and front right arc (-scan_arc_rad to 0)
            valid_mask = np.isfinite(ranges) & (ranges > 0.05) & (ranges < 3.0)
            
            left_mask = valid_mask & (angles >= 0.0) & (angles <= self.scan_arc_rad)
            right_mask = valid_mask & (angles >= -self.scan_arc_rad) & (angles < 0.0)
            
            left_ranges = ranges[left_mask] if np.any(left_mask) else np.array([999.0])
            right_ranges = ranges[right_mask] if np.any(right_mask) else np.array([999.0])
            
            self.min_left_dist = float(np.min(left_ranges))
            self.min_right_dist = float(np.min(right_ranges))
            
            min_front_dist = min(self.min_left_dist, self.min_right_dist)
            
            # Calculate continuous evasive offset
            if min_front_dist < self.alert_distance:
                self.obstacle_detected = True
                # Scaling factor: closer obstacle -> higher offset
                intensity = 1.0 - (min_front_dist / self.alert_distance)
                scaled_offset = self.max_offset * intensity
                
                # Steer AWAY from the closer obstacle side
                if self.min_left_dist < self.min_right_dist:
                    self.current_offset = +scaled_offset  # Steer Right
                else:
                    self.current_offset = -scaled_offset  # Steer Left
            else:
                self.obstacle_detected = False
                self.current_offset = 0.0
                
        except Exception:
            pass
        finally:
            try:
                self._lock.release()
            except RuntimeError:
                pass

    def get_offset(self):
        """
        Returns calculated LiDAR evasive offset (-0.35 .. +0.35)
        """
        return self.current_offset
