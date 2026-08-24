#!/usr/bin/env python3
"""
SmartCity App Runner - Điều khiển xe JetRacer tự hành trong Sa hình Thành phố Thông minh.
"""

import sys
import os
import time
import requests
from enum import Enum
import cv2
import numpy as np

try:
    import rospy
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

from config.settings import SMART_CITY_VIDEO_OUTPUT
from jetracer_ai.core.base_runner import BaseRacerNode


class RobotState(Enum):
    WAITING_FOR_LINE = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    HANDLING_EVENT = 3
    LEAVING_INTERSECTION = 4
    REACQUIRING_LINE = 5
    DEAD_END = 6
    GOAL_REACHED = 7


class SmartCityController(BaseRacerNode):
    """Bộ điều khiển bài thi Smart City."""

    def __init__(self):
        super().__init__(node_name="jetracer_smart_city", video_filename=SMART_CITY_VIDEO_OUTPUT)
        self.setup_parameters()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)

    def setup_parameters(self):
        self.ROI_Y = int(self.height * 0.85)
        self.ROI_H = int(self.height * 0.15)
        self.LINE_COLOR_LOWER = np.array([0, 0, 0])
        self.LINE_COLOR_UPPER = np.array([180, 255, 75])
        self.INTERSECTION_CLEARANCE_DURATION = 1.5
        self.INTERSECTION_APPROACH_DURATION = 0.5
        self.RF_MODEL = "dataset3-c4kyj"
        self.RF_VERSION = "1"
        self.RF_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')

    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            if not initial and HAS_ROS:
                rospy.loginfo(f"State: {self.current_state.name if self.current_state else 'NONE'} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time() if HAS_ROS else time.time()

    def _get_line_center(self, image):
        if image is None:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        roi_mask = mask[self.ROI_Y:self.ROI_Y + self.ROI_H, :]
        M = cv2.moments(roi_mask)
        if M["m00"] > 100:
            return int(M["m10"] / M["m00"])
        return None

    def correct_course(self, line_center_x):
        if line_center_x is None:
            return
        center_x = self.width / 2.0
        error = (line_center_x - center_x) / center_x
        steering = error * 0.8
        self.car_controller.steer(steering, 0.20)

    def process_frame(self, frame):
        debug_info = {
            "State": self.current_state.name if self.current_state else "UNKNOWN"
        }
        return self.draw_debug_overlay(frame, debug_info)

    def run(self):
        if HAS_ROS:
            rospy.loginfo("Smart City Runner starting...")
            rate = rospy.Rate(20)
            while not rospy.is_shutdown():
                if self.latest_image is not None:
                    line_c = self._get_line_center(self.latest_image)
                    if self.current_state == RobotState.WAITING_FOR_LINE:
                        if line_c is not None:
                            self._set_state(RobotState.DRIVING_STRAIGHT)
                    elif self.current_state == RobotState.DRIVING_STRAIGHT:
                        if self.detector.process_detection():
                            self.car_controller.stop()
                        elif line_c is not None:
                            self.correct_course(line_c)
                rate.sleep()


if __name__ == "__main__":
    controller = SmartCityController()
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.stop()
