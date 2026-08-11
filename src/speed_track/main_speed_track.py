#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import rospy
import cv2
import numpy as np
import time
import os
import json
import math
from enum import Enum
import requests

from src.core.control.racer_controller import RacerController
import onnxruntime as ort
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
from sensor_msgs.msg import LaserScan, Image
from src.core.utils.opposite_detector import SimpleOppositeDetector

from src.core.planning.map_navigator import MapNavigator

class RobotState(Enum):
    WAITING_FOR_LINE = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    HANDLING_EVENT = 3
    LEAVING_INTERSECTION = 4
    REACQUIRING_LINE = 5
    DEAD_END = 6
    GOAL_REACHED = 7

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

class JetRacerController:
    def __init__(self):
        rospy.loginfo("Đang khởi tạo JetRacer Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_mqtt()

        self.video_writer = None
        self.initialize_video_writer()

        self.navigator = MapNavigator(self.MAP_FILE_PATH)
        self.current_node_id = self.navigator.start_node
        self.target_node_id = None
        self.planned_path = None
        self.banned_edges = []
        self.plan_initial_route()

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký vào các topic /scan và /csi_cam_0/image_raw.")
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
        rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

    def plan_initial_route(self): 
        """Lập kế hoạch đường đi ban đầu từ điểm xuất phát đến đích."""
        rospy.loginfo(f"Đang lập kế hoạch từ node {self.navigator.start_node} đến {self.navigator.end_node}...")
        self.planned_path = self.navigator.find_path(
            self.navigator.start_node, 
            self.navigator.end_node,
            self.banned_edges
        )
        if self.planned_path and len(self.planned_path) > 1:
            self.target_node_id = self.planned_path[1]
            rospy.loginfo(f"Đã tìm thấy đường đi: {self.planned_path}. Đích đến đầu tiên: {self.target_node_id}")
        else:
            rospy.logerr("Không tìm thấy đường đi hoặc đường đi quá ngắn!")
            self._set_state(RobotState.DEAD_END)

    def initialize_video_writer(self):
        """Khởi tạo đối tượng VideoWriter."""
        try:
            # Kích thước video sẽ giống kích thước ảnh robot xử lý
            frame_size = (self.WIDTH, self.HEIGHT)
            self.video_writer = cv2.VideoWriter(self.VIDEO_OUTPUT_FILENAME, 
                                                self.VIDEO_FOURCC, 
                                                self.VIDEO_FPS, 
                                                frame_size)
            if self.video_writer.isOpened():
                rospy.loginfo(f"Bắt đầu ghi video vào file '{self.VIDEO_OUTPUT_FILENAME}'")
            else:
                rospy.logerr("Không thể mở file video để ghi.")
                self.video_writer = None
        except Exception as e:
            rospy.logerr(f"Lỗi khi khởi tạo VideoWriter: {e}")
            self.video_writer = None

    def draw_debug_info(self, image):
        """Vẽ các thông tin gỡ lỗi lên một khung hình."""
        if image is None:
            return None
        
        debug_frame = image.copy()
        
        # 1. Vẽ các ROI
        # ROI Chính (màu xanh lá)
        cv2.rectangle(debug_frame, (0, self.ROI_Y), (self.WIDTH-1, self.ROI_Y + self.ROI_H), (0, 255, 0), 1)
        # ROI Dự báo (màu vàng)
        cv2.rectangle(debug_frame, (0, self.LOOKAHEAD_ROI_Y), (self.WIDTH-1, self.LOOKAHEAD_ROI_Y + self.LOOKAHEAD_ROI_H), (0, 255, 255), 1)

        # 2. Vẽ trạng thái hiện tại
        state_text = f"State: {self.current_state.name}"
        cv2.putText(debug_frame, state_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 3. Vẽ line và trọng tâm (nếu robot đang bám line)
        if self.current_state == RobotState.DRIVING_STRAIGHT:
            # Lấy line center của ROI Chính
            line_center = self._get_line_center(image, self.ROI_Y, self.ROI_H)
            if line_center is not None:
                # Vẽ một đường thẳng đứng màu đỏ tại vị trí trọng tâm
                cv2.line(debug_frame, (line_center, self.ROI_Y), (line_center, self.ROI_Y + self.ROI_H), (0, 0, 255), 2)

        return debug_frame

    def setup_parameters(self):
        self.WIDTH, self.HEIGHT = 300, 300
        self.ROI_Y = int(self.HEIGHT * 0.85)
        self.ROI_H = int(self.HEIGHT * 0.15)
        self.ROI_CENTER_WIDTH_PERCENT = 0.5
        self.LOOKAHEAD_ROI_Y = int(self.HEIGHT * 0.60)
        self.LOOKAHEAD_ROI_H = int(self.HEIGHT * 0.15)

        self.LINE_COLOR_LOWER = np.array([0, 0, 0])
        self.LINE_COLOR_UPPER = np.array([180, 255, 75])
        self.INTERSECTION_CLEARANCE_DURATION = 1.5
        self.INTERSECTION_APPROACH_DURATION = 0.5
        self.LINE_REACQUIRE_TIMEOUT = 3.0
        self.SCAN_PIXEL_THRESHOLD = 100
        self.YOLO_MODEL_PATH = "models/best.onnx"
        self.YOLO_CONF_THRESHOLD = 0.6
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['N', 'E', 'W', 'S', 'NN', 'NE', 'NW', 'NS', 'math']
        self.PRESCRIPTIVE_SIGNS = {'N', 'E', 'W', 'S'}
        self.PROHIBITIVE_SIGNS = {'NN', 'NE', 'NW', 'NS'}
        self.DATA_ITEMS = {'qr_code', 'math_problem'}
        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"
        self.current_state = None
        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        self.ANGLE_TO_FACE_SIGN_MAP = {d: a for d, a in zip(self.DIRECTIONS, [45, -45, -135, 135])}
        self.MAP_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "core", "utils", "map.json")
        self.MIN_DETECTION_CONFIRM_TIME = 1.5
        self.LABEL_TO_DIRECTION_ENUM = {'N': Direction.NORTH, 'E': Direction.EAST, 'S': Direction.SOUTH, 'W': Direction.WEST}
        self.VIDEO_OUTPUT_FILENAME = 'jetracer_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

    def initialize_hardware(self):
        """Khởi tạo JetRacer thông qua RacerController."""
        self.controller = RacerController()
        rospy.loginfo("JetRacer hardware đã được khởi tạo qua RacerController.")

    def initialize_yolo(self):
        """Tải mô hình YOLO vào ONNX Runtime."""
        try:
            self.yolo_session = ort.InferenceSession(self.YOLO_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            rospy.loginfo("Tải mô hình YOLO thành công.")
        except Exception as e:
            rospy.logerr(f"Không thể tải mô hình YOLO từ '{self.YOLO_MODEL_PATH}'. Lỗi: {e}")
            self.yolo_session = None

    def numpy_nms(self, boxes, scores, iou_threshold):
        """
        Thực hiện Non-Maximum Suppression (NMS) bằng NumPy.
        :param boxes: list các bounding box, mỗi box là [x1, y1, x2, y2]
        :param scores: list các điểm tin cậy tương ứng
        :param iou_threshold: ngưỡng IoU để loại bỏ các box trùng lặp
        :return: list các chỉ số (indices) của các box được giữ lại
        """
        # Chuyển đổi sang NumPy array để tính toán vector hóa
        x1 = np.array([b[0] for b in boxes])
        y1 = np.array([b[1] for b in boxes])
        x2 = np.array([b[2] for b in boxes])
        y2 = np.array([b[3] for b in boxes])

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # Sắp xếp các box theo điểm tin cậy giảm dần
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Tính toán IoU (Intersection over Union)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            # Giữ lại các box có IoU nhỏ hơn ngưỡng
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def detect_with_yolo(self, image):
        """
        Thực hiện nhận diện đối tượng bằng YOLOv8 và hậu xử lý kết quả đúng cách.
        """
        if self.yolo_session is None: return []

        original_height, original_width = image.shape[:2]

        img_resized = cv2.resize(image, self.YOLO_INPUT_SIZE)
        img_data = np.array(img_resized, dtype=np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(img_data, axis=0)  # Add batch dimension

        input_name = self.yolo_session.get_inputs()[0].name
        outputs = self.yolo_session.run(None, {input_name: input_tensor})

        # Lấy output thô, output của YOLOv8 thường có shape (1, 84, 8400) hoặc tương tự
        # Chúng ta cần transpose nó thành (1, 8400, 84) để dễ xử lý
        predictions = np.squeeze(outputs[0]).T

        # Lọc các box có điểm tin cậy (objectness score) thấp
        # Cột 4 trong predictions là điểm tin cậy tổng thể của box
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.YOLO_CONF_THRESHOLD, :]
        scores = scores[scores > self.YOLO_CONF_THRESHOLD]

        if predictions.shape[0] == 0:
            rospy.loginfo("YOLO không phát hiện đối tượng nào vượt ngưỡng tin cậy.")
            return []

        # Lấy class_id có điểm cao nhất
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Lấy tọa độ box và chuyển đổi về ảnh gốc
        x, y, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        
        # Tính toán tỷ lệ scale để chuyển đổi tọa độ
        x_scale = original_width / self.YOLO_INPUT_SIZE[0]
        y_scale = original_height / self.YOLO_INPUT_SIZE[1]

        # Chuyển từ [center_x, center_y, width, height] sang [x1, y1, x2, y2]
        x1 = (x - w / 2) * x_scale
        y1 = (y - h / 2) * y_scale
        x2 = (x + w / 2) * x_scale
        y2 = (y + h / 2) * y_scale
        
        # Chuyển thành list các box và scores
        boxes = np.column_stack((x1, y1, x2, y2)).tolist()
        
        # 4. Thực hiện Non-Maximum Suppression (NMS)
        # Đây là một bước cực kỳ quan trọng để loại bỏ các box trùng lặp
        # OpenCV cung cấp một hàm NMS hiệu quả
        nms_threshold = 0.45 # Ngưỡng IOU để loại bỏ box
        indices = self.numpy_nms(np.array(boxes), scores, nms_threshold)
        
        if len(indices) == 0:
            rospy.loginfo("YOLO: Sau NMS, không còn đối tượng nào.")
            return []

        # 5. Tạo danh sách kết quả cuối cùng
        final_detections = []
        for i in indices.flatten():
            final_detections.append({
                'class_name': self.YOLO_CLASS_NAMES[class_ids[i]],
                'confidence': float(scores[i]),
                'box': [int(coord) for coord in boxes[i]] # Chuyển tọa độ sang int
            })

        rospy.loginfo(f"YOLO đã phát hiện {len(final_detections)} đối tượng cuối cùng.")
        return final_detections

    def initialize_mqtt(self):
        self.mqtt_client = mqtt.Client()
        def on_connect(client, userdata, flags, rc): rospy.loginfo(f"Kết nối MQTT: {'Thành công' if rc == 0 else 'Thất bại'}")
        self.mqtt_client.on_connect = on_connect
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e: rospy.logerr(f"Không thể kết nối MQTT: {e}")
    
    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            if not initial: rospy.loginfo(f"Chuyển trạng thái: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    def camera_callback(self, image_msg):
        try:
            if image_msg.encoding.endswith('compressed'):
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
            if 'rgb' in image_msg.encoding: cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
        except Exception as e: rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    def run(self):
        rospy.loginfo("Bắt đầu vòng lặp. Đợi 3 giây...") 
        time.sleep(3) 
        rospy.loginfo("Hành trình bắt đầu!")
        self.detector.start_scanning()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            # ===================================================================
            # TRẠNG THÁI 0: ĐANG CHỜ TÌM THẤY LINE (WAITING_FOR_LINE)
            # ===================================================================
            if self.current_state == RobotState.WAITING_FOR_LINE:
                rospy.loginfo_throttle(5, "Đang ở trạng thái chờ... Tìm kiếm vạch kẻ đường để bắt đầu.")
                
                # Giữ robot đứng yên
                self.controller.stop()

                # Kiểm tra xem đã có ảnh chưa
                if self.latest_image is None:
                    rospy.logwarn_throttle(2, "Chưa nhận được hình ảnh camera. Hãy đảm bảo gym-donkeycar.py và simulator đang chạy.")
                    rate.sleep()
                    continue
                
                # Kiểm tra xem line có xuất hiện trong cả hai ROI không để đảm bảo ổn định
                lookahead_line = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                execution_line = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)

                if lookahead_line is not None and execution_line is not None:
                    rospy.loginfo("Đã tìm thấy vạch kẻ đường! Bắt đầu hành trình.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)
                    # Không cần continue, để vòng lặp tiếp theo tự nhiên chuyển sang DRIVING_STRAIGHT
                
                # (Tùy chọn: Thêm timeout nếu muốn)
                # if rospy.get_time() - self.state_change_time > 30: # Ví dụ timeout 30 giây
                #     rospy.logerr("Timeout! Không tìm thấy line để bắt đầu.")
                #     self._set_state(RobotState.DEAD_END)
            # ===================================================================
            # TRẠNG THÁI 1: ĐANG BÁM LINE (DRIVING_STRAIGHT)
            # ===================================================================
            if self.current_state == RobotState.DRIVING_STRAIGHT:
                if self.latest_image is None:
                    rospy.logwarn_throttle(5, "Đang chờ dữ liệu hình ảnh từ topic camera...")
                    self.controller.stop()
                    rate.sleep()
                    continue

                # --- BƯỚC 1: KIỂM TRA TÍN HIỆU ƯU TIÊN CAO (LiDAR) ---
                # Đây là tín hiệu đáng tin cậy nhất, nếu nó kích hoạt, xử lý ngay.
                # Chặn các phát hiện LiDAR quá sớm ngay sau khi chuyển sang DRIVING_FORTH để tránh
                # nhận diện nhầm node bắt đầu là một giao lộ mới.
                time_since_state = rospy.get_time() - self.state_change_time
                if time_since_state >= self.MIN_DETECTION_CONFIRM_TIME and self.detector.process_detection():
                    rospy.loginfo("SỰ KIỆN (LiDAR): Phát hiện giao lộ. Dừng ngay lập tức.")
                    self.controller.stop()
                    time.sleep(0.5) # Chờ robot dừng hẳn

                    # Cập nhật vị trí hiện tại (đã đến đích) và xử lý
                    self.current_node_id = self.target_node_id
                    rospy.loginfo(f"==> ĐÃ ĐẾN node {self.current_node_id}.")

                    if self.current_node_id == self.navigator.end_node:
                        rospy.loginfo("ĐÃ ĐẾN ĐÍCH CUỐI CÙNG!")
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()
                    continue # Bắt đầu vòng lặp mới với trạng thái mới

                # --- BƯỚC 2: LOGIC "NHÌN XA HƠN" VỚI ROI DỰ BÁO ---
                # Nếu LiDAR im lặng, kiểm tra xem vạch kẻ có sắp biến mất ở phía xa không.
                lookahead_line_center = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)

                if lookahead_line_center is None:
                    rospy.logwarn("SỰ KIỆN (Dự báo): Vạch kẻ đường biến mất ở phía xa. Chuẩn bị vào giao lộ.")
                    # Hành động phòng ngừa: chuyển sang trạng thái đi thẳng vào giao lộ.
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue # Bắt đầu vòng lặp mới với trạng thái mới

                # --- BƯỚC 3: BÁM LINE BÌNH THƯỜNG (NẾU PHÍA TRƯỚC AN TOÀN) ---
                # Chỉ khi cả LiDAR và ROI Dự báo đều ổn, ta mới thực hiện bám line.
                execution_line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)

                if execution_line_center is not None:
                    # An toàn để bám line, vì chúng ta biết phía trước không có giao lộ đột ngột.
                    self.correct_course(execution_line_center)
                    self.controller.reset_pid()
                else:
                    # Trường hợp hiếm: ROI xa thấy line nhưng ROI gần lại không. Dừng lại cho an toàn.
                    rospy.logwarn("Trạng thái không nhất quán: ROI xa thấy line, ROI gần không thấy. Tạm dừng an toàn.")
                    self.controller.stop()

            # ===================================================================
            # TRẠNG THÁI 2: ĐANG TIẾN VÀO GIAO LỘ (APPROACHING_INTERSECTION)
            # ===================================================================
            elif self.current_state == RobotState.APPROACHING_INTERSECTION:
                # Đi thẳng một đoạn ngắn để vào trung tâm giao lộ
                self.controller.forward()
                
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_APPROACH_DURATION:
                    rospy.loginfo("Đã tiến vào trung tâm giao lộ. Dừng lại để xử lý.")
                    self.controller.stop()
                    time.sleep(0.5)

                    self.current_node_id = self.target_node_id
                    rospy.loginfo(f"==> ĐÃ ĐẾN node {self.current_node_id}.")

                    if self.current_node_id == self.navigator.end_node:
                        rospy.loginfo("ĐÃ ĐẾN ĐÍCH CUỐI CÙNG!")
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()

            # ===================================================================
            # TRẠNG THÁI 3: ĐANG RỜI KHỎI GIAO LỘ (LEAVING_INTERSECTION)
            # ===================================================================
            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.controller.forward()
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_CLEARANCE_DURATION:
                    rospy.loginfo("Đã thoát khỏi khu vực giao lộ. Bắt đầu tìm kiếm line mới.")
                    self._set_state(RobotState.REACQUIRING_LINE)
            
            # ===================================================================
            # TRẠNG THÁI 4: ĐANG TÌM LẠI LINE (REACQUIRING_LINE)
            # ===================================================================
            elif self.current_state == RobotState.REACQUIRING_LINE:
                self.controller.forward()
                line_center_x = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                
                if line_center_x is not None:
                    rospy.loginfo("Đã tìm thấy line mới! Chuyển sang chế độ bám line.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)
                    continue
                
                if rospy.get_time() - self.state_change_time > self.LINE_REACQUIRE_TIMEOUT:
                    rospy.logerr("Không thể tìm thấy line mới sau khi rời giao lộ. Dừng lại.")
                    self._set_state(RobotState.DEAD_END)

            # ===================================================================
            # TRẠNG THÁI KẾT THÚC (DEAD_END, GOAL_REACHED)
            # ===================================================================
            elif self.current_state == RobotState.DEAD_END:
                rospy.logwarn("Đã vào ngõ cụt hoặc gặp lỗi không thể phục hồi. Dừng hoạt động.")
                self.controller.stop()
                break
            elif self.current_state == RobotState.GOAL_REACHED: 
                rospy.loginfo("ĐÃ HOÀN THÀNH NHIỆM VỤ. Dừng hoạt động.")
                self.controller.stop()
                break

            self._record_frame()

            rate.sleep()
        self.cleanup()

    def _record_frame(self):
        """Hàm trợ giúp để vẽ thông tin và ghi một khung hình vào video."""
        if self.video_writer is not None and self.latest_image is not None:
            debug_frame = self.draw_debug_info(self.latest_image)
            if debug_frame is not None:
                self.video_writer.write(debug_frame)

    def cleanup(self):
        rospy.loginfo("Dừng robot và giải phóng tài nguyên...")
        if hasattr(self, 'controller') and self.controller is not None:
            self.controller.stop()

        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo("Đã lưu và đóng file video.")
        
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.stop_scanning()
            
        if hasattr(self, 'mqtt_client') and self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
        rospy.loginfo("Đã giải phóng tài nguyên. Chương trình kết thúc.")

    def map_absolute_to_relative(self, target_direction_label, current_robot_direction):
        """
        Chuyển đổi hướng tuyệt đối ('N', 'E', 'S', 'W') thành hành động tương đối ('straight', 'left', 'right').
        Ví dụ: robot đang hướng BẮC (NORTH), mục tiêu là đi hướng ĐÔNG (EAST) -> hành động là 'right'.
        """
        target_dir = self.LABEL_TO_DIRECTION_ENUM.get(target_direction_label)
        if target_dir is None: return None

        current_idx = current_robot_direction.value
        target_idx = target_dir.value
        
        diff = (target_idx - current_idx + 4) % 4 
        
        if diff == 0:
            return 'straight'
        elif diff == 1:
            return 'right'
        elif diff == 3: 
            return 'left'
        else: 
            return 'turn_around'
        
    def map_relative_to_absolute(self, relative_action, current_robot_direction):
        """
        Chuyển đổi hành động tương đối ('straight', 'left', 'right') thành hướng tuyệt đối ('N', 'E', 'S', 'W').
        """
        current_idx = current_robot_direction.value
        if relative_action == 'straight':
            target_idx = current_idx
        elif relative_action == 'right':
            target_idx = (current_idx + 1) % 4
        elif relative_action == 'left':
            target_idx = (current_idx - 1 + 4) % 4
        else:
            return None
        
        for label, direction in self.LABEL_TO_DIRECTION_ENUM.items():
            if direction.value == target_idx:
                return label
        return None
    
    def _get_line_center(self, image, roi_y, roi_h):
        """Kiểm tra sự tồn tại và vị trí của vạch kẻ trong một ROI cụ thể."""
        if image is None: return None
        roi = image[roi_y : roi_y + roi_h, :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Bước 1: Tạo mặt nạ màu sắc như cũ
        color_mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        
        # === BƯỚC 2: TẠO MẶT NẠ TẬP TRUNG (FOCUS MASK) ===
        focus_mask = np.zeros_like(color_mask)
        roi_height, roi_width = focus_mask.shape
        
        center_width = int(roi_width * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (roi_width - center_width) // 2
        end_x = start_x + center_width
        
        # Vẽ một hình chữ nhật trắng ở giữa
        cv2.rectangle(focus_mask, (start_x, 0), (end_x, roi_height), 255, -1)
        
        # === BƯỚC 3: KẾT HỢP HAI MẶT NẠ ===
        # Chỉ giữ lại những pixel trắng nào xuất hiện ở cả hai mặt nạ
        final_mask = cv2.bitwise_and(color_mask, focus_mask)

        # (Tùy chọn) Hiển thị mask để debug
        # cv2.imshow("Color Mask", color_mask)
        # cv2.imshow("Focus Mask", focus_mask)
        # cv2.imshow("Final Mask", final_mask)
        # cv2.waitKey(1)
        
        # Tìm contours trên mặt nạ cuối cùng đã được lọc
        _, contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None
            
        c = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(c) < self.SCAN_PIXEL_THRESHOLD:
            return None

        M = cv2.moments(c)
        if M["m00"] > 0:
            # Quan trọng: Trọng tâm bây giờ được tính toán chỉ dựa trên vạch kẻ trong khu vực trung tâm
            return int(M["m10"] / M["m00"])
        return None
    
    def correct_course(self, line_center_x):
        """
        Bám line bằng PID Controller (JetRacer Ackermann steering).
        Thay vì chênh lệch tốc độ 2 motor (JetBot), dùng góc lái servo.
        """
        error = line_center_x - (self.WIDTH / 2)
        self.controller.correct_course_pid(error, self.WIDTH)
        
    def handle_intersection(self):
        rospy.loginfo("\n[GIAO LỘ] Dừng lại và xử lý theo bản đồ (không quét)...")
        self.controller.stop()
        time.sleep(0.2)

        # Cập nhật vị trí hiện tại
        self.current_node_id = self.target_node_id
        rospy.loginfo(f"==> ĐÃ ĐẾN node {self.current_node_id}.")

        if self.current_node_id == self.navigator.end_node:
            rospy.loginfo("ĐÃ ĐẾN ĐÍCH CUỐI CÙNG!")
            self._set_state(RobotState.GOAL_REACHED)
            return

        rospy.loginfo("Lập kế hoạch tiếp theo theo map (không dùng scan/YOLO)...")
        planned_direction_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
        if not planned_direction_label:
            rospy.logerr("Lỗi kế hoạch: Không tìm thấy bước tiếp theo.")
            self._set_state(RobotState.DEAD_END)
            return

        planned_action = self.map_absolute_to_relative(planned_direction_label, self.DIRECTIONS[self.current_direction_index])
        rospy.loginfo(f"Kế hoạch đề xuất: Đi {planned_action} (hướng {planned_direction_label})")

        # Thực hiện hành động theo kế hoạch (không quét, không ưu tiên biển báo)
        if planned_action == 'straight':
            rospy.loginfo("[FINAL] Decision: Go STRAIGHT.")
            # nothing to do; continue forward in LEAVING_INTERSECTION
        elif planned_action == 'right':
            rospy.loginfo("[FINAL] Decision: Turn RIGHT.")
            self.turn_robot(90, True)
        elif planned_action == 'left':
            rospy.loginfo("[FINAL] Decision: Turn LEFT.")
            self.turn_robot(-90, True)
        elif planned_action == 'turn_around':
            rospy.loginfo("[FINAL] Decision: Turn AROUND.")
            self.turn_robot(180, True)
        else:
            rospy.logwarn("[!!!] DEAD END! No valid paths found.")
            self._set_state(RobotState.DEAD_END)
            return

        # Sau khi quay để thực hiện rẽ, cố gắng ổn định heading nhỏ để dễ bắt lại vạch
        self.stabilize_after_turn()

        # Xác định node tiếp theo theo đường đi đã lập
        try:
            next_node_id = self.planned_path[self.planned_path.index(self.current_node_id) + 1]
        except Exception:
            rospy.logerr("Không thể xác định node tiếp theo từ kế hoạch.")
            self._set_state(RobotState.DEAD_END)
            return

        self.target_node_id = next_node_id
        rospy.loginfo(f"==> Đang di chuyển đến node tiếp theo: {self.target_node_id}")
        self._set_state(RobotState.LEAVING_INTERSECTION)

    def stabilize_after_turn(self, max_attempts=2, small_angle=6):
        """Try small in-place sweeps to help the camera/vision reacquire the line after a turn.

        Strategy:
        - Do small negative rotation (e.g. -6deg) and check if path exists in ROI.
        - Return to center, then try small positive rotation.
        - Repeat a couple of times. Do not update main heading while sweeping.
        """
        if self.latest_image is None:
            return False

        # Quick check: maybe line already exists in frame
        if self._does_path_exist_in_frame(self.latest_image):
            return True

        for attempt in range(max_attempts):
            # try left small
            self.turn_robot(-small_angle, update_main_direction=False)
            time.sleep(0.05)
            if self.latest_image is not None and self._does_path_exist_in_frame(self.latest_image):
                # found on left sweep
                # return robot to nominal heading
                self.turn_robot(small_angle, update_main_direction=False)
                return True
            # return to center
            self.turn_robot(small_angle, update_main_direction=False)

            # try right small
            self.turn_robot(small_angle, update_main_direction=False)
            time.sleep(0.05)
            if self.latest_image is not None and self._does_path_exist_in_frame(self.latest_image):
                # found on right sweep
                self.turn_robot(-small_angle, update_main_direction=False)
                return True
            # return to center
            self.turn_robot(-small_angle, update_main_direction=False)

        # nothing found
        return False
    
    def turn_robot(self, degrees, update_main_direction=True):
        """Rẽ JetRacer bằng vòng cung (KHÔNG quay tại chỗ)."""
        self.controller.turn_angle(degrees, record_callback=self._record_frame)
        if update_main_direction and degrees % 90 == 0 and degrees != 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns + 4) % 4
            rospy.loginfo(f"==> Hướng đi MỚI: {self.DIRECTIONS[self.current_direction_index].name}")
    
    def _does_path_exist_in_frame(self, image):
        if image is None: return False
        roi = image[self.ROI_Y : self.ROI_Y + self.ROI_H, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        _img, contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return bool(contours) and cv2.contourArea(max(contours, key=cv2.contourArea)) > self.SCAN_PIXEL_THRESHOLD
    
    def scan_for_available_paths_proactive(self):
        rospy.loginfo("[SCAN] Bắt đầu quét chủ động...")
        paths = {"straight": False, "right": False, "left": False}
        if self.latest_image is not None:
            paths["straight"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False)
        time.sleep(0.5)
        if self.latest_image is not None:
            paths["right"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(-180, update_main_direction=False)
        time.sleep(0.5)
        if self.latest_image is not None:
            paths["left"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False)
        rospy.loginfo(f"[SCAN] Kết quả: {paths}")
        return paths

def main():
    rospy.init_node('jetbot_controller_node', anonymous=True)
    try:
        controller = JetRacerController()
        controller.run()
    except rospy.ROSInterruptException: rospy.loginfo("Node đã bị ngắt.")
    except Exception as e: rospy.logerr(f"Lỗi không xác định: {e}", exc_info=True)

if __name__ == '__main__':
    main()