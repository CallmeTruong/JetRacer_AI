#!/usr/bin/env python3
"""
SimpleOppositeDetector - Thut ton pht hin vt th i din bng LiDAR Scan.
"""

import time
import math
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import LaserScan
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


class SimpleOppositeDetector:
    """(see module docstring)"""

    def __init__(self):
        # (translated)
        self.min_distance = 0.25
        self.max_distance = 0.35
        self.object_min_points = 15
        self.distance_threshold = 0.10
        self.angle_range = 20.0
        self.detection_interval = 2.0
        self.opposite_tolerance = 5.0
        self.min_opposite_distance = 45.0

        # (translated)
        self.scanning_active = True
        self.last_detection_time = 0
        self.latest_scan = None

    def start_scanning(self):
        """(see module docstring)"""
        self.scanning_active = True
        return True

    def stop_scanning(self):
        """(see module docstring)"""
        self.scanning_active = False
        self.latest_scan = None
        return True

    def callback(self, scan):
        """(see module docstring)"""
        if not self.scanning_active:
            return
        self.latest_scan = scan
        current_time = time.time()
        if current_time - self.last_detection_time >= self.detection_interval:
            self.last_detection_time = current_time

    def index_to_angle(self, index, scan):
        """(see module docstring)"""
        angle_rad = scan.angle_min + (index * scan.angle_increment)
        return math.degrees(angle_rad)

    def get_angle_difference(self, angle1, angle2):
        """(see module docstring)"""
        diff = abs(angle1 - angle2)
        return 360.0 - diff if diff > 180.0 else diff

    def are_opposite(self, angle1, angle2):
        """(see module docstring)"""
        return abs(self.get_angle_difference(angle1, angle2) - 180.0) <= self.opposite_tolerance

    def find_all_objects(self, scan):
        """(see module docstring)"""
        ranges = np.array(scan.ranges)
        n = len(ranges)
        if n == 0 or scan.angle_increment == 0:
            return []

        angle_increment_deg = math.degrees(scan.angle_increment)
        points_per_range = max(1, int(self.angle_range / angle_increment_deg))
        objects = []

        for start_idx in range(0, n, max(1, points_per_range // 2)):
            end_idx = min(start_idx + points_per_range, n)
            if end_idx - start_idx < points_per_range // 2:
                continue
            zone_ranges = ranges[start_idx:end_idx]
            center_idx = start_idx + (end_idx - start_idx) // 2
            center_angle = self.index_to_angle(center_idx, scan)
            obj = self.detect_object_in_zone(zone_ranges, f"Zone_{start_idx}")
            if obj:
                obj['center_angle'] = center_angle
                obj['center_index'] = center_idx
                objects.append(obj)

        return objects

    def find_opposite_pairs(self, objects):
        """(see module docstring)"""
        opposite_pairs = []
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1:]:
                angle_diff = self.get_angle_difference(obj1['center_angle'], obj2['center_angle'])
                if angle_diff >= self.min_opposite_distance and self.are_opposite(obj1['center_angle'], obj2['center_angle']):
                    opposite_pairs.append({
                        'object1': obj1,
                        'object2': obj2,
                        'angle_difference': angle_diff
                    })
        return opposite_pairs

    def process_detection(self):
        """(see module docstring)"""
        if self.latest_scan is None:
            return False
        scan = self.latest_scan
        all_objects = self.find_all_objects(scan)
        if len(all_objects) < 2:
            return False

        opposite_pairs = self.find_opposite_pairs(all_objects)
        if opposite_pairs:
            opposite_pairs.sort(key=lambda x: abs(x['angle_difference'] - 180.0))
            return True
        return False

    def detect_object_in_zone(self, zone_ranges, zone_name):
        """(see module docstring)"""
        if len(zone_ranges) == 0:
            return None
        valid_mask = (zone_ranges >= self.min_distance) & (zone_ranges <= self.max_distance) & np.isfinite(zone_ranges)
        if np.sum(valid_mask) < self.object_min_points:
            return None

        valid_ranges = zone_ranges[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        clusters, current_cluster = [], [0]

        for i in range(1, len(valid_ranges)):
            if (valid_indices[i] - valid_indices[current_cluster[-1]] <= 2 and
                    abs(valid_ranges[i] - valid_ranges[current_cluster[-1]]) <= self.distance_threshold):
                current_cluster.append(i)
            else:
                if len(current_cluster) >= self.object_min_points:
                    clusters.append(current_cluster)
                current_cluster = [i]

        if len(current_cluster) >= self.object_min_points:
            clusters.append(current_cluster)

        if not clusters:
            return None

        largest_cluster = max(clusters, key=len)
        cluster_distances = [valid_ranges[i] for i in largest_cluster]
        return {
            'distance': float(np.mean(cluster_distances)),
            'point_count': len(largest_cluster),
            'zone': zone_name
        }