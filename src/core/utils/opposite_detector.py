#!/usr/bin/env python
# opposite_detector_node.py

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool, SetBoolResponse
import time
import json
import math

class SimpleOppositeDetector:
    def __init__(self):
        # Các tham số cố định
        self.min_distance = 0.25
        self.max_distance = 0.35
        self.object_min_points = 15
        self.distance_threshold = 0.10
        self.angle_range = 20.0
        self.detection_interval = 2.0
        self.opposite_tolerance = 5.0
        self.min_opposite_distance = 45.0
        
        # Biến điều khiển
        self.scanning_active = False
        self.subscriber = None
        self.last_detection_time = 0
        self.latest_scan = None
        
        # Publisher để gửi tín hiệu giao lộ
#         self.notification_pub = rospy.Publisher('/intersection_trigger', String, queue_size=10)
#         self.status_pub = rospy.Publisher('/scan_status', Bool, queue_size=1)
        
        # Service để bật/tắt quét
#         self.start_service = rospy.Service('/start_scanning', SetBool, self.start_scanning_service)
#         self.stop_service = rospy.Service('/stop_scanning', SetBool, self.stop_scanning_service)
        
#         rospy.loginfo("Opposite Detector initialized.")
    
#     def start_scanning_service(self, req):
#         if req.data: return self.start_scanning()
#         else: return self.stop_scanning()
    
#     def stop_scanning_service(self, req):
#         if req.data: return self.stop_scanning()
#         else: return self.start_scanning()
    
    def start_scanning(self):
        try:
            if not self.scanning_active:
                self.subscriber = rospy.Subscriber('/scan', LaserScan, self.callback)
                self.scanning_active = True
#                 self.status_pub.publish(True)
#                 rospy.loginfo("Scanning STARTED")
                return SetBoolResponse(success=True, message="Scanning started")
            return SetBoolResponse(success=True, message="Already active")
        except Exception as e:
            rospy.logerr(f"Failed to start scanning: {e}")
            return SetBoolResponse(success=False, message=f"Failed to start: {e}")
    
    def stop_scanning(self):
        try:
            if self.scanning_active:
                if self.subscriber:
                    self.subscriber.unregister()
                    self.subscriber = None
                self.scanning_active = False
                self.latest_scan = None
#                 self.status_pub.publish(False)
                rospy.loginfo("Scanning STOPPED")
                return SetBoolResponse(success=True, message="Scanning stopped")
            return SetBoolResponse(success=True, message="Already inactive")
        except Exception as e:
            rospy.logerr(f"Failed to stop scanning: {e}")
            return SetBoolResponse(success=False, message=f"Failed to stop: {e}")

    def callback(self, scan):
        if not self.scanning_active: return
        self.latest_scan = scan
        current_time = time.time()
        if current_time - self.last_detection_time >= self.detection_interval:
            # self.process_detection()
            self.last_detection_time = current_time
    
    def index_to_angle(self, index, scan):
        angle_rad = scan.angle_min + (index * scan.angle_increment)
        return math.degrees(angle_rad)
    
    def get_angle_difference(self, angle1, angle2):
        diff = abs(angle1 - angle2)
        return 360 - diff if diff > 180 else diff
    
    def are_opposite(self, angle1, angle2):
        return abs(self.get_angle_difference(angle1, angle2) - 180.0) <= self.opposite_tolerance
    
    def find_all_objects(self, scan):
        # (logic phát hiện vật thể)
        ranges = np.array(scan.ranges)
        n = len(ranges)
        angle_increment_deg = math.degrees(scan.angle_increment)
        points_per_range = int(self.angle_range / angle_increment_deg)
        objects = []
        for start_idx in range(0, n, points_per_range // 2):
            end_idx = min(start_idx + points_per_range, n)
            if end_idx - start_idx < points_per_range // 2: continue
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
        opposite_pairs = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                angle_diff = self.get_angle_difference(obj1['center_angle'], obj2['center_angle'])
                if angle_diff >= self.min_opposite_distance and self.are_opposite(obj1['center_angle'], obj2['center_angle']):
                    opposite_pairs.append({'object1': obj1, 'object2': obj2, 'angle_difference': angle_diff})
        return opposite_pairs
    
    def process_detection(self):
        if self.latest_scan is None: return
        scan = self.latest_scan
        timestamp = rospy.get_time()
        all_objects = self.find_all_objects(scan)
        if len(all_objects) < 2: return
        
        opposite_pairs = self.find_opposite_pairs(all_objects)
        
        if opposite_pairs:
            opposite_pairs.sort(key=lambda x: abs(x['angle_difference'] - 180.0))
            best_pair = opposite_pairs[0]
            # rospy.loginfo("[%.1f] *** OPPOSITE OBJECTS DETECTED ***", timestamp)
            # Tạo và gửi tin nhắn
            notification = {
                "timestamp": timestamp,
                "detection_type": 'OPPOSITE_OBJECTS',
                # ... thông tin chi tiết khác
            }
#             self.notification_pub.publish(json.dumps(notification))
            return True
        else:
            # rospy.loginfo("[%.1f] Found %d objects, but none are opposite", timestamp, len(all_objects))
            return False

    def detect_object_in_zone(self, zone_ranges, zone_name):
        if len(zone_ranges) == 0: return None
        valid_mask = (zone_ranges >= self.min_distance) & (zone_ranges <= self.max_distance) & np.isfinite(zone_ranges)
        if np.sum(valid_mask) < self.object_min_points: return None
        valid_ranges = zone_ranges[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        clusters, current_cluster = [], [0]
        for i in range(1, len(valid_ranges)):
            if (valid_indices[i] - valid_indices[current_cluster[-1]] <= 2 and abs(valid_ranges[i] - valid_ranges[current_cluster[-1]]) <= self.distance_threshold):
                current_cluster.append(i)
            else:
                if len(current_cluster) >= self.object_min_points: clusters.append(current_cluster)
                current_cluster = [i]
        if len(current_cluster) >= self.object_min_points: clusters.append(current_cluster)
        if not clusters: return None
        largest_cluster = max(clusters, key=len)
        cluster_distances = [valid_ranges[i] for i in largest_cluster]
        return {'distance': np.mean(cluster_distances), 'point_count': len(largest_cluster), 'zone': zone_name}

# if __name__ == '__main__':
#     rospy.init_node('opposite_detector_node', anonymous=True)
#     detector = SimpleOppositeDetector()
#     rospy.loginfo("Opposite Detector Node is running. Use services to control.")
#     rospy.spin()