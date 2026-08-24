#!/usr/bin/env python3
"""
SpeedTrack App Runner - Điều khiển xe JetRacer đua tốc độ bám làn.
"""

import sys
import os
import time
from enum import Enum
import cv2
import numpy as np

try:
    import rospy
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

from config.settings import SPEED_TRACK_VIDEO_OUTPUT
from jetracer_ai.core.base_runner import BaseRacerNode


class TrackState(Enum):
    WAITING_FOR_LINE = 0
    RACING = 1
    EVADING_OBSTACLE = 2
    FINISH = 3


class SpeedTrackController(BaseRacerNode):
    """Bộ điều khiển tối ưu tốc độ đua."""

    def __init__(self):
        super().__init__(node_name="jetracer_speed_track", video_filename=SPEED_TRACK_VIDEO_OUTPUT)
        self.setup_parameters()
        self.state = TrackState.WAITING_FOR_LINE

    def setup_parameters(self):
        self.ROI_Y = int(self.height * 0.70)
        self.ROI_H = int(self.height * 0.25)
        self.RACE_THROTTLE = 0.35
        self.CORNER_THROTTLE = 0.20
        self.LINE_COLOR_LOWER = np.array([0, 0, 0])
        self.LINE_COLOR_UPPER = np.array([180, 255, 75])

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

    def race_control(self, line_center_x):
        if line_center_x is None:
            self.car_controller.forward(self.CORNER_THROTTLE)
            return

        center_x = self.width / 2.0
        error = (line_center_x - center_x) / center_x
        steering = error * 1.0

        throttle = self.CORNER_THROTTLE if abs(steering) > 0.4 else self.RACE_THROTTLE
        self.car_controller.steer(steering, throttle)

    def process_frame(self, frame):
        debug_info = {
            "Race State": self.state.name,
            "Throttle": f"{self.RACE_THROTTLE:.2f}",
        }
        return self.draw_debug_overlay(frame, debug_info)

    def run(self):
        if HAS_ROS:
            rospy.loginfo("Speed Track Runner starting...")
            rate = rospy.Rate(30)
            while not rospy.is_shutdown():
                if self.latest_image is not None:
                    line_center = self._get_line_center(self.latest_image)
                    if self.state == TrackState.WAITING_FOR_LINE:
                        if line_center is not None:
                            self.state = TrackState.RACING
                    elif self.state == TrackState.RACING:
                        if self.detector.process_detection():
                            self.car_controller.forward(0.10)
                        else:
                            self.race_control(line_center)
                rate.sleep()


if __name__ == "__main__":
    controller = SpeedTrackController()
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.stop()
