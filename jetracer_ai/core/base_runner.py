"""
BaseRacerNode - Lp c s tru tng cho tt c cc runner ca JetRacer (Smart City, Speed Track).
Gom ton b x l giao tip ROS, Camera, LiDAR, MQTT, Video Recording v Debug Drawing.
"""

import sys
import os
import time
import json
import cv2
import numpy as np
from enum import Enum

# (translated)
sys.path = [p for p in sys.path if 'python2.7' not in p]

try:
    import rospy
    from sensor_msgs.msg import LaserScan, Image
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

from config.settings import (
    IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_FPS, VIDEO_FOURCC,
    ROS_TOPIC_CAM, ROS_TOPIC_SCAN,
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC_STATUS, MQTT_TOPIC_MAP_LOAD,
    MAP_JSON_PATH
)
from jetracer_ai.core.racer_controller import RacerController
from jetracer_ai.core.opposite_detector import SimpleOppositeDetector



class BaseRacerNode:
    """(see module docstring)"""

    def __init__(self, node_name="jetracer_node", video_filename="output.mp4"):
        self.node_name = node_name
        self.video_filename = video_filename

        # (translated)
        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT

        # (translated)
        self.latest_image = None
        self.latest_scan = None
        self.video_writer = None
        self.mqtt_client = None

        if HAS_ROS:
            try:
                rospy.init_node(self.node_name, anonymous=True)
                rospy.loginfo(f"Khi to ROS Node '{self.node_name}'...")
            except rospy.ROSException:
                pass

        # (translated)
        self.car_controller = RacerController()

        # (translated)
        self.detector = SimpleOppositeDetector()

        # (translated)

        self.initialize_video_writer()

        # (translated)
        self.initialize_mqtt()

        # (translated)
        if HAS_ROS:
            rospy.Subscriber(ROS_TOPIC_SCAN, LaserScan, self.detector.callback)
            rospy.Subscriber(ROS_TOPIC_CAM, Image, self.camera_callback)
            rospy.loginfo(f" ng k cc topic: {ROS_TOPIC_SCAN}, {ROS_TOPIC_CAM}")

    def initialize_video_writer(self):
        """(see module docstring)"""
        try:
            frame_size = (self.width, self.height)
            self.video_writer = cv2.VideoWriter(
                self.video_filename,
                VIDEO_FOURCC,
                VIDEO_FPS,
                frame_size
            )
            if self.video_writer.isOpened():
                if HAS_ROS:
                    rospy.loginfo(f"Bt u ghi video vo file '{self.video_filename}'")
                else:
                    print(f"[+] Recording video to '{self.video_filename}'")
            else:
                self.video_writer = None
        except Exception as e:
            if HAS_ROS:
                rospy.logerr(f"VideoWriter initialization error: {e}")
            self.video_writer = None

    def initialize_mqtt(self):
        """(see module docstring)"""
        if not HAS_MQTT:
            return
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            if HAS_ROS:
                rospy.logwarn(f"Khng th kt ni MQTT broker ({MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}): {e}")

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0 and HAS_ROS:
            rospy.loginfo(" kt ni MQTT Broker thnh cng.")
            client.subscribe(MQTT_TOPIC_MAP_LOAD)

    def _on_mqtt_message(self, client, userdata, msg):
        """(see module docstring)"""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            self.handle_mqtt_payload(msg.topic, payload)
        except Exception as e:
            if HAS_ROS:
                rospy.logerr(f"Li parse MQTT message: {e}")

    def handle_mqtt_payload(self, topic, payload):
        """(see module docstring)"""
        pass

    def publish_mqtt_status(self, data_dict):
        """(see module docstring)"""
        if self.mqtt_client and HAS_MQTT:
            try:
                payload = json.dumps(data_dict)
                self.mqtt_client.publish(MQTT_TOPIC_STATUS, payload)
            except Exception as e:
                if HAS_ROS:
                    rospy.logwarn(f"Li gi MQTT status: {e}")

    def camera_callback(self, msg):
        """(see module docstring)"""
        try:
            # Decode NumPy BGR image
            im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding in ['rgb8', 'RGB8']:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ['rgba8', 'RGBA8']:
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            elif msg.encoding in ['bgra8', 'BGRA8']:
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

            if im.shape[0] != self.height or im.shape[1] != self.width:
                im = cv2.resize(im, (self.width, self.height))

            self.latest_image = im

            # (translated)
            processed_frame = self.process_frame(im)

            if processed_frame is None:
                processed_frame = im

            # Ghi video
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.write(processed_frame)

        except Exception as e:
            if HAS_ROS:
                rospy.logerr(f"Li camera callback: {e}")

    def process_frame(self, frame):
        """(see module docstring)"""
        return frame

    def draw_debug_overlay(self, image, info_dict):
        """(see module docstring)"""
        debug_img = image.copy()
        y_offset = 20
        for key, val in info_dict.items():
            text = f"{key}: {val}"
            cv2.putText(
                debug_img, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            y_offset += 20
        return debug_img

    def stop(self):
        """(see module docstring)"""
        if self.car_controller:
            self.car_controller.stop()
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
        if self.mqtt_client and HAS_MQTT:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        if HAS_ROS:
            rospy.loginfo(f"Node '{self.node_name}'  dng an ton.")
