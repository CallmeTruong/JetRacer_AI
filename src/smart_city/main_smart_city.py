# ...existing code...
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
import importlib.util
from datetime import datetime, timezone

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
        rospy.loginfo("Đang khởi tạo JetRacer Smart City Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_yolo()  # now sets up Roboflow API parameters
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
        # Use shortest-through-loads if available in MapNavigator; fallback to find_path
        if hasattr(self.navigator, 'find_shortest_path_through_loads'):
            self.planned_path = self.navigator.find_shortest_path_through_loads(
                self.navigator.start_node,
                self.navigator.end_node,
                self.banned_edges
            )
        else:
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
        # YOLO fields remain for compatibility but we will use Roboflow API
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
        self.LABEL_TO_DIRECTION_ENUM = {'N': Direction.NORTH, 'E': Direction.EAST, 'S': Direction.SOUTH, 'W': Direction.WEST}
        self.VIDEO_OUTPUT_FILENAME = 'jetracer_smartcity_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

        # --- Roboflow integration config ---
        # Model identifier given by user: dataset3-c4kyj/1
        self.RF_MODEL = "dataset3-c4kyj"
        self.RF_VERSION = "1"
        # Prefer environment var ROBOFLOW_API_KEY, fallback to empty (user should set)
        self.RF_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')
        # submission script path (may contain space in filename per user)
        self.SUBMIT_SCRIPT_FILENAME = "submit_sign copy.py"
        self.SUBMIT_URL = os.environ.get('SUBMIT_URL', 'http://example.com/submit')  # fallback if submit script not present
        self.TEAM_NAME = os.environ.get('TEAM_NAME', 'Tên đội thi')

    def initialize_hardware(self):
        """Khởi tạo JetRacer thông qua RacerController."""
        self.controller = RacerController()
        rospy.loginfo("JetRacer hardware đã được khởi tạo qua RacerController.")

    def initialize_yolo(self):
        """Setup for Roboflow API (used instead of local YOLO ONNX model)."""
        # We keep YOLO session variable for backward compatibility but we won't load ONNX here.
        self.yolo_session = None
        # Attempt to locate submit script
        self.submit_module = self._load_submit_module()

    def _load_submit_module(self):
        """Try to import local 'submit_sign copy.py' and expose a submit function if available."""
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, self.SUBMIT_SCRIPT_FILENAME)
        if not os.path.exists(path):
            rospy.loginfo(f"Submit script not found at {path}, will fallback to HTTP submit to {self.SUBMIT_URL}")
            return None
        try:
            spec = importlib.util.spec_from_file_location("submit_sign_copy", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            rospy.loginfo("Loaded submit script module from '%s'." % path)
            return module
        except Exception as e:
            rospy.logerr(f"Không thể load submit script '{path}': {e}")
            return None

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
        Replaced YOLO local detection by Roboflow detection API.
        Returns list in same format as previous detect_with_yolo:
        [{'class_name': str, 'confidence': float, 'box': [x1,y1,x2,y2]}, ...]
        """
        if image is None:
            return []

        if not self.RF_API_KEY:
            rospy.logerr("Roboflow API key not set (ROBOFLOW_API_KEY). Skipping detection.")
            return []

        try:
            # encode image as jpeg
            _, img_encoded = cv2.imencode('.jpg', image)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            # endpoint: https://detect.roboflow.com/{model}/{version}
            url = f"https://detect.roboflow.com/{self.RF_MODEL}/{self.RF_VERSION}"
            params = {'api_key': self.RF_API_KEY}
            resp = requests.post(url, params=params, files=files, timeout=8)
            if resp.status_code != 200:
                rospy.logerr(f"Roboflow detect failed: {resp.status_code} {resp.text}")
                return []
            j = resp.json()
            preds = j.get('predictions', [])
            detections = []
            h, w = image.shape[:2]
            for p in preds:
                # Roboflow returns x,y,width,height in pixels or relative depending on model;
                # handle both numeric types
                cx = p.get('x')
                cy = p.get('y')
                pw = p.get('width')
                ph = p.get('height')
                # If values are relative (0..1), scale to pixels
                if 0 < cx <= 1 and 0 < cy <= 1 and 0 < pw <= 1 and 0 < ph <= 1:
                    cx *= w; cy *= h; pw *= w; ph *= h
                x1 = int(cx - pw/2); y1 = int(cy - ph/2); x2 = int(cx + pw/2); y2 = int(cy + ph/2)
                detections.append({
                    'class_name': str(p.get('class', p.get('label', 'unknown'))),
                    'confidence': float(p.get('confidence', 0.0)),
                    'box': [max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)]
                })
            # NEW: print detections to ROS log for on-screen/debug visibility
            try:
                rospy.loginfo(f"Roboflow detections: {json.dumps(detections, ensure_ascii=False)}")
            except Exception:
                rospy.loginfo(f"Roboflow detections: {detections}")
            rospy.loginfo(f"Roboflow detected {len(detections)} objects.")
            return detections
        except Exception as e:
            rospy.logerr(f"Error during Roboflow detection: {e}")
            return []

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
                    rate.sleep()
                    continue
                
                # Kiểm tra xem line có xuất hiện trong cả hai ROI không để đảm bảo ổn định
                lookahead_line = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                execution_line = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)

                if lookahead_line is not None and execution_line is not None:
                    rospy.loginfo("Đã tìm thấy vạch kẻ đường! Bắt đầu hành trình.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)
                    # Không cần continue, để vòng lặp tiếp theo tự nhiên chuyển sang DRIVING_STRAIGHT
                
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
                if self.detector.process_detection():
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
                lookahead_line_center = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)

                if lookahead_line_center is None:
                    rospy.logwarn("SỰ KIỆN (Dự báo): Vạch kẻ đường biến mất ở phía xa. Chuẩn bị vào giao lộ.")
                    # Hành động phòng ngừa: chuyển sang trạng thái đi thẳng vào giao lộ.
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue # Bắt đầu vòng lặp mới với trạng thái mới

                # --- BƯỚC 3: BÁM LINE BÌNH THƯỜNG (NẾU PHÍA TRƯỚC AN TOÀN) ---
                execution_line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)

                if execution_line_center is not None:
                    # An toàn để bám line, vì chúng ta biết phía trước không có giao lộ đột ngột.
                    self.correct_course(execution_line_center)
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
        
    def _submit_payload(self, payload):
        """
        Use the local submit script if available; otherwise perform HTTP POST to SUBMIT_URL.
        Expect server to return JSON with status code 201 on success.
        """
        try:
            if self.submit_module is not None:
                # Try common function names in submit script
                for fn in ('submit', 'submit_detection', 'main'):
                    if hasattr(self.submit_module, fn):
                        func = getattr(self.submit_module, fn)
                        try:
                            resp = func(payload)
                            rospy.loginfo(f"Submit script returned: {resp}")
                            return resp
                        except Exception as e:
                            rospy.logwarn(f"Call to submit script function '{fn}' failed: {e}")
                rospy.logwarn("Submit module loaded but no callable submit function found; falling back to HTTP POST.")
            # fallback HTTP POST
            headers = {'Content-Type': 'application/json'}
            r = requests.post(self.SUBMIT_URL, json=payload, headers=headers, timeout=6)
            try:
                j = r.json()
            except Exception:
                j = {'text': r.text}
            result = {'status_code': r.status_code, 'response': j}
            rospy.loginfo(f"HTTP submit result: {result}")
            return result
        except Exception as e:
            rospy.logerr(f"Error submitting payload: {e}")
            return {'status_code': 0, 'response': str(e)}

    def handle_intersection(self):
        rospy.loginfo("\n[GIAO LỘ] Dừng lại và xử lý...")
        self.controller.stop() 
        time.sleep(0.5)

        current_direction = self.DIRECTIONS[self.current_direction_index]
        angle_to_sign = self.ANGLE_TO_FACE_SIGN_MAP.get(current_direction, 0)
        self.turn_robot(angle_to_sign, False)
        image_info = self.latest_image
        detections = self.detect_with_yolo(image_info)
        self.turn_robot(-angle_to_sign, False)
        
        prescriptive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PRESCRIPTIVE_SIGNS}
        prohibitive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PROHIBITIVE_SIGNS}
        data_items = [det for det in detections if det['class_name'] in self.DATA_ITEMS]

        # 2. Xử lý các mục dữ liệu (QR, Toán) và Publish
        rospy.loginfo("[STEP 2] Processing data items...")
        for item in data_items:
            if item['class_name'] == 'qr_code':
                rospy.loginfo("Found QR Code. Publishing data...")
                self.publish_data({'type': 'QR_CODE', 'value': 'simulated_data_123'})
            elif item['class_name'] == 'math_problem':
                rospy.loginfo("Found Math Problem. Solving and publishing...")
                self.publish_data({'type': 'MATH_PROBLEM', 'value': '2+2=4'})

        # Attempt to submit detected signs (prescriptive/prohibitive) to server via submit module or HTTP
        for det in detections:
            try:
                payload = {
                    "text": f"Detected: {det.get('class_name')}, conf={det.get('confidence')}",
                    "race": 1,
                    "node_id": int(self.current_node_id) if isinstance(self.current_node_id, (int,str)) and str(self.current_node_id).isdigit() else self.current_node_id,
                    "submit_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z'),
                    "team": self.TEAM_NAME,
                    "Map_type": self.MAP_FILE_PATH
                }
                rospy.loginfo(f"Submitting detection payload: {payload}")
                submit_result = self._submit_payload(payload)
                # If HTTP fallback returned structure, map to expected short log
                status = submit_result.get('status_code') if isinstance(submit_result, dict) else None
                if status == 201:
                    rospy.loginfo(f"Thành công (Status Code: 201) -> {submit_result.get('response')}")
                else:
                    rospy.loginfo(f"Submit returned: {submit_result}")
            except Exception as e:
                rospy.logerr(f"Error while submitting detection: {e}")

        rospy.loginfo("[STEP 3] Lập kế hoạch điều hướng theo bản đồ...")
        # 3. Lập kế hoạch Điều hướng
        final_decision = None
        is_deviation = False 

        while True:
            planned_direction_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
            if not planned_direction_label:
                rospy.logerr("Lỗi kế hoạch: Không tìm thấy bước tiếp theo.") 
                self._set_state(RobotState.DEAD_END) 
                return
            
            planned_action = self.map_absolute_to_relative(planned_direction_label, current_direction)
            rospy.loginfo(f"Kế hoạch A* đề xuất: Đi {planned_action} (hướng {planned_direction_label})")

            # Ưu tiên 1: Biển báo bắt buộc
            intended_action = None
            if 'L' in prescriptive_cmds: intended_action = 'left'
            elif 'R' in prescriptive_cmds: intended_action = 'right'
            elif 'F' in prescriptive_cmds: intended_action = 'straight'
            
            # Ưu tiên 2: Plan
            if intended_action is None:
                intended_action = planned_action
            else:
                # Nếu hành động bắt buộc khác với kế hoạch, đánh dấu là đi chệch hướng
                if intended_action != planned_action:
                    is_deviation = True
                    rospy.logwarn(f"CHỆCH HƯỚNG! Biển báo bắt buộc ({intended_action}) khác với kế hoạch ({planned_action}).")

            # 3.3. Veto bởi biển báo cấm
            is_prohibited = (intended_action == 'straight' and 'NF' in prohibitive_cmds) or \
                            (intended_action == 'right' and 'NR' in prohibitive_cmds) or \
                            (intended_action == 'left' and 'NL' in prohibitive_cmds)

            if is_prohibited:
                rospy.logwarn(f"Hành động dự định '{intended_action}' bị CẤM!")
                
                # Nếu hành động bị cấm đến từ biển báo bắt buộc -> Lỗi bản đồ
                if is_deviation:
                    rospy.logerr("LỖI BẢN ĐỒ! Biển báo bắt buộc mâu thuẫn với biển báo cấm. Không thể đi tiếp.")
                    self._set_state(RobotState.DEAD_END) 
                    return
                
                # Nếu hành động bị cấm đến từ kế hoạch A* -> Tìm đường lại
                banned_edge = (self.current_node_id, self.planned_path[self.planned_path.index(self.current_node_id) + 1])
                if banned_edge not in self.banned_edges:
                    self.banned_edges.append(banned_edge)
                
                rospy.loginfo(f"Thêm cạnh cấm {banned_edge} và tìm đường lại...")
                # Use find_shortest_path_through_loads if available to respect loads requirement
                if hasattr(self.navigator, 'find_shortest_path_through_loads'):
                    new_path = self.navigator.find_shortest_path_through_loads(self.current_node_id, self.navigator.end_node, self.banned_edges)
                else:
                    new_path = self.navigator.find_path(self.current_node_id, self.navigator.end_node, self.banned_edges)

                if new_path:
                    self.planned_path = new_path
                    rospy.loginfo(f"Đã tìm thấy đường đi mới: {self.planned_path}")
                    continue # Quay lại đầu vòng lặp để kiểm tra với kế hoạch mới
                else:
                    rospy.logerr("Không thể tìm đường đi mới sau khi gặp biển cấm.")
                    self._set_state(RobotState.DEAD_END)
                    return
            
            final_decision = intended_action
            break 

        # 4. Thực thi quyết định
        if final_decision == 'straight': 
            rospy.loginfo("[FINAL] Decision: Go STRAIGHT.")
        elif final_decision == 'right': 
            rospy.loginfo("[FINAL] Decision: Turn RIGHT.") 
            self.turn_robot(90, True)
        elif final_decision == 'left': 
            rospy.loginfo("[FINAL] Decision: Turn LEFT.") 
            self.turn_robot(-90, True)
        else:
            rospy.logwarn("[!!!] DEAD END! No valid paths found.") 
            self._set_state(RobotState.DEAD_END)
            return
        
        # 5. Cập nhật trạng thái robot sau khi thực hiện
        # 5.1. Xác định node tiếp theo
        next_node_id = None
        if not is_deviation:
            # Nếu đi theo kế hoạch, chỉ cần lấy node tiếp theo từ path
            next_node_id = self.planned_path[self.planned_path.index(self.current_node_id) + 1]
        else:
            # Nếu chệch hướng, phải tìm node tiếp theo dựa trên hành động đã thực hiện
            new_robot_direction = self.DIRECTIONS[self.current_direction_index] 
            
            executed_direction_label = None
            for label, direction_enum in self.LABEL_TO_DIRECTION_ENUM.items():
                if direction_enum == new_robot_direction:
                    executed_direction_label = label 
                    break
            
            if executed_direction_label is None:
                rospy.logerr("Lỗi logic: Không thể tìm thấy label cho hướng đi mới của robot.") 
                self._set_state(RobotState.DEAD_END) 
                return

            next_node_id = self.navigator.get_neighbor_by_direction(self.current_node_id, executed_direction_label)
            if next_node_id is None:
                 rospy.logerr("LỖI BẢN ĐỒ! Đã thực hiện rẽ nhưng không có node tương ứng.")
                 self._set_state(RobotState.DEAD_END)
                 return
            
            # Quan trọng: Lập kế hoạch lại từ vị trí mới
            rospy.loginfo(f"Đã đi chệch kế hoạch. Lập lại đường đi từ node mới {next_node_id}...")
            if hasattr(self.navigator, 'find_shortest_path_through_loads'):
                new_path = self.navigator.find_shortest_path_through_loads(next_node_id, self.navigator.end_node, self.banned_edges)
            else:
                new_path = self.navigator.find_path(next_node_id, self.navigator.end_node, self.banned_edges)
            if new_path:
                self.planned_path = new_path
                rospy.loginfo(f"Đường đi mới sau khi chệch hướng: {self.planned_path}")
            else:
                rospy.logerr("Không thể tìm đường về đích từ vị trí mới.")
                self._set_state(RobotState.DEAD_END)
                return

        self.target_node_id = next_node_id
        rospy.loginfo(f"==> Đang di chuyển đến node tiếp theo: {self.target_node_id}")
        self._set_state(RobotState.LEAVING_INTERSECTION)
    
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
        controller = JetBotController()
        controller.run()
    except rospy.ROSInterruptException: rospy.loginfo("Node đã bị ngắt.")
    except Exception as e: rospy.logerr(f"Lỗi không xác định: {e}", exc_info=True)

if __name__ == '__main__':
    main()
# ...existing code...