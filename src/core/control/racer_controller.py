#!/usr/bin/env python3
"""
RacerController - Lớp trừu tượng điều khiển JetRacer (Ackermann Steering)

Xe JetRacer dùng Ackermann steering (lái servo phía trước + motor ga phía sau),
KHÔNG PHẢI differential drive như JetBot.

Khác biệt chính:
  - JetBot:    robot.set_motors(left_speed, right_speed)  → quay tại chỗ được
  - JetRacer:  car.steering = angle, car.throttle = speed → KHÔNG quay tại chỗ

Lớp này cung cấp API thống nhất để code chính không cần quan tâm loại xe.

Cách dùng:
    from src.core.control.racer_controller import RacerController
    controller = RacerController()
    controller.forward(0.3)          # Đi thẳng
    controller.steer(0.5, 0.3)       # Rẽ phải nhẹ + đi tới
    controller.turn_angle(90)        # Rẽ phải 90 độ (đi vòng cung)
    controller.stop()                # Dừng
"""

import sys
# Lọc bỏ đường dẫn python2.7 của ROS Melodic để không đè lên các thư viện Python 3
sys.path = [p for p in sys.path if 'python2.7' not in p]

import time
import math

try:
    import rospy
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

# ============================================================
# CẤU HÌNH PHẦN CỨNG - ĐIỀU CHỈNH THEO XE THẬT
# ============================================================
DEFAULT_CONFIG = {
    # --- Throttle (ga) ---
    "BASE_THROTTLE": 0.20,         # Tốc độ đi thẳng mặc định (0.0 → 1.0)
    "TURN_THROTTLE": 0.15,         # Tốc độ khi đang rẽ (chậm hơn)
    "MAX_THROTTLE": 0.40,          # Giới hạn tốc độ tối đa (an toàn)

    # --- Steering (lái) ---
    "STEERING_GAIN": 0.8,          # Hệ số khuếch đại góc lái khi bám line
    "MAX_STEERING": 1.0,           # Giới hạn góc lái tối đa (-1.0 trái ↔ +1.0 phải)
    "STEERING_OFFSET": 0.0,        # Bù lệch nếu xe bị lệch (calibrate trên xe thật)

    # --- Timing (thời gian rẽ) ---
    "TURN_DURATION_90_DEG": 1.5,   # Thời gian (giây) để rẽ 90° ở TURN_THROTTLE
    "STEERING_VALUE_FOR_TURN": 0.7, # Giá trị steering khi rẽ gấp (0.0→1.0)

    # --- PID Controller ---
    "PID_KP": 0.5,                 # Proportional gain
    "PID_KI": 0.0,                 # Integral gain (thường để 0 cho robot nhỏ)
    "PID_KD": 0.1,                 # Derivative gain (giảm dao động)

    # --- Safety ---
    "SAFE_ZONE_PERCENT": 0.3,      # Vùng an toàn ở giữa (% chiều rộng ảnh)
}


class RacerController:
    """
    Controller cho JetRacer (Waveshare JetRacer Pro AI Kit)
    
    Hardware:
    - Servo lái (steering): PCA9685 channel, range -1.0 (trái) → +1.0 (phải)
    - Motor ga (throttle):  PCA9685 channel, range -1.0 (lùi) → +1.0 (tiến)
    - RPLIDAR trên nóc
    - CSI Camera phía trước
    """

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.car = None
        self._mock = False

        # PID state
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = time.time()

        self._initialize_hardware()

    def _initialize_hardware(self):
        """Khởi tạo phần cứng JetRacer."""
        # Thử import jetracer (thư viện chính thức NVIDIA)
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self.car = NvidiaRacecar()
            self.car.steering = 0.0
            self.car.throttle = 0.0
            self._log("Khởi tạo JetRacer (NvidiaRacecar) thành công.")
            return
        except Exception as e:
            self._log(f"Không tìm thấy jetracer library: {e}", level="warn")

        # Thử import jetbot_pro (Waveshare variant)
        try:
            from jetbot import Robot
            self.car = Robot()
            self._mock = False
            self._log("Khởi tạo JetBot Pro (fallback) thành công.")
            self._log("LƯU Ý: Đang dùng JetBot API trên JetRacer - cần kiểm tra tương thích!", level="warn")
            return
        except Exception as e:
            self._log(f"Không tìm thấy jetbot library: {e}", level="warn")

        # Mock mode (cho dev/test trên laptop)
        self._log("Không tìm thấy phần cứng → Chạy ở chế độ MÔ PHỎNG (Mock).", level="warn")
        from unittest.mock import Mock
        self.car = Mock()
        self._mock = True

    # ============================================================
    # API CHÍNH - Điều khiển cơ bản
    # ============================================================

    def forward(self, speed=None):
        """Đi thẳng với tốc độ cho trước."""
        speed = speed or self.config["BASE_THROTTLE"]
        speed = self._clamp_throttle(speed)
        self._set_steering(0.0)
        self._set_throttle(speed)

    def stop(self):
        """Dừng robot ngay lập tức."""
        self._set_throttle(0.0)
        self._set_steering(0.0)

    def steer(self, steering_value, speed=None):
        """
        Đi với góc lái cho trước.
        
        Args:
            steering_value: -1.0 (trái max) → 0.0 (thẳng) → +1.0 (phải max)
            speed: tốc độ, mặc định BASE_THROTTLE
        """
        speed = speed or self.config["BASE_THROTTLE"]
        speed = self._clamp_throttle(speed)
        steering_value = self._clamp_steering(steering_value)
        self._set_steering(steering_value)
        self._set_throttle(speed)

    # ============================================================
    # API NÂNG CAO - Rẽ góc, PID bám line
    # ============================================================

    def turn_angle(self, degrees, record_callback=None):
        """
        Rẽ một góc cho trước (đi vòng cung, KHÔNG quay tại chỗ).
        
        JetRacer không thể quay tại chỗ như JetBot, nên phải:
        1. Đánh lái sang một bên
        2. Đi tới với tốc độ chậm
        3. Đợi đủ thời gian
        4. Trả lái thẳng + dừng

        Args:
            degrees: góc rẽ (dương = phải, âm = trái)
            record_callback: hàm ghi video debug (gọi mỗi frame)
        """
        if degrees == 0:
            return

        # Tính thời gian rẽ tỷ lệ với góc
        duration = abs(degrees) / 90.0 * self.config["TURN_DURATION_90_DEG"]
        turn_steering = self.config["STEERING_VALUE_FOR_TURN"]
        turn_throttle = self.config["TURN_THROTTLE"]

        # Xác định hướng lái
        if degrees > 0:
            self._set_steering(turn_steering)      # Lái phải
        else:
            self._set_steering(-turn_steering)     # Lái trái

        # Đi tới trong khi đánh lái
        self._set_throttle(turn_throttle)

        # Đợi đủ thời gian (ghi video debug nếu có)
        start_time = time.time()
        while time.time() - start_time < duration:
            if record_callback:
                record_callback()
            time.sleep(0.05)  # 20 FPS

        # Dừng + trả lái thẳng
        self.stop()
        time.sleep(0.3)

        if record_callback:
            record_callback()

    def correct_course_pid(self, error, image_width):
        """
        PID Controller cho bám line.
        
        Thay thế P-controller cũ bằng PID đầy đủ.
        Đầu ra là giá trị steering (-1.0 → +1.0) thay vì chênh lệch tốc độ motor.
        
        Args:
            error: sai lệch pixel từ tâm ảnh (dương = lệch phải)
            image_width: chiều rộng ảnh (pixels)
        """
        # Chuẩn hóa error về range [-1, 1]
        normalized_error = error / (image_width / 2.0)

        # Kiểm tra vùng an toàn (dead zone)
        if abs(normalized_error) < self.config["SAFE_ZONE_PERCENT"]:
            self.forward()
            return

        # PID calculation
        current_time = time.time()
        dt = current_time - self._pid_last_time
        if dt <= 0:
            dt = 0.05  # fallback

        # P - Proportional
        p_term = self.config["PID_KP"] * normalized_error

        # I - Integral (với anti-windup)
        self._pid_integral += normalized_error * dt
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral))  # clamp
        i_term = self.config["PID_KI"] * self._pid_integral

        # D - Derivative
        d_error = (normalized_error - self._pid_last_error) / dt
        d_term = self.config["PID_KD"] * d_error

        # Tổng hợp
        steering_output = p_term + i_term + d_term
        steering_output = self._clamp_steering(steering_output)

        # Cập nhật state
        self._pid_last_error = normalized_error
        self._pid_last_time = current_time

        # Áp dụng
        self.steer(steering_output, self.config["BASE_THROTTLE"])

    def reset_pid(self):
        """Reset PID state (gọi khi chuyển trạng thái)."""
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = time.time()

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _set_throttle(self, value):
        """Set throttle, tương thích cả NvidiaRacecar và JetBot."""
        value = self._clamp_throttle(value)
        if hasattr(self.car, 'throttle'):
            # NvidiaRacecar API
            self.car.throttle = value
        elif hasattr(self.car, 'set_motors'):
            # JetBot API fallback - cả 2 motor cùng tốc
            self.car.set_motors(value, value)
        elif hasattr(self.car, 'forward'):
            if value > 0:
                self.car.forward(value)
            elif value < 0:
                self.car.backward(abs(value))
            else:
                self.car.stop()

        if self._mock and HAS_ROS:
            self.current_throttle = value
            if not hasattr(self, 'current_steering'):
                self.current_steering = 0.0
            if hasattr(rospy, 'publish_control'):
                try:
                    rospy.publish_control(self.current_steering, self.current_throttle)
                except Exception:
                    pass

    def _set_steering(self, value):
        """Set steering, tương thích cả NvidiaRacecar và JetBot."""
        value = self._clamp_steering(value)
        value += self.config["STEERING_OFFSET"]
        if hasattr(self.car, 'steering'):
            # NvidiaRacecar API
            self.car.steering = value
        elif hasattr(self.car, 'set_motors') and not self._mock:
            # JetBot API fallback - dùng chênh lệch tốc độ
            # Lưu ý: Cách này chỉ hoạt động tương đối trên JetBot, 
            # KHÔNG phản ánh đúng Ackermann steering
            pass  # Steering sẽ được xử lý trong _set_throttle

        if self._mock and HAS_ROS:
            self.current_steering = value
            if not hasattr(self, 'current_throttle'):
                self.current_throttle = 0.0
            if hasattr(rospy, 'publish_control'):
                try:
                    rospy.publish_control(self.current_steering, self.current_throttle)
                except Exception:
                    pass

    def _clamp_throttle(self, value):
        return max(-self.config["MAX_THROTTLE"], min(self.config["MAX_THROTTLE"], value))

    def _clamp_steering(self, value):
        return max(-self.config["MAX_STEERING"], min(self.config["MAX_STEERING"], value))

    def _log(self, msg, level="info"):
        if HAS_ROS:
            if level == "warn":
                rospy.logwarn(msg)
            elif level == "error":
                rospy.logerr(msg)
            else:
                rospy.loginfo(msg)
        else:
            print(f"[{level.upper()}] {msg}")


# ============================================================
# TEST (chạy trực tiếp file này để test)
# ============================================================
if __name__ == "__main__":
    print("=== Test RacerController ===")
    ctrl = RacerController()
    print("Forward...")
    ctrl.forward(0.2)
    time.sleep(1)
    print("Steer right...")
    ctrl.steer(0.5, 0.2)
    time.sleep(1)
    print("Turn 90 degrees...")
    ctrl.turn_angle(90)
    print("Stop.")
    ctrl.stop()
    print("=== Test complete ===")
