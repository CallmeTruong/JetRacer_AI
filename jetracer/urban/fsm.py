import time
from jetracer.urban.config import ROUTE_COMMANDS

class UrbanFSMPlanner:
    """
    Finite State Machine (FSM) High-Level Decision Planner for Urban Autonomous Driving.
    
    States:
        - STATE_DRIVING           : Bình thường - Xe di chuyển theo đường
        - STATE_STOPPED_RED_LIGHT : Dừng đèn đỏ (Bắt buộc dừng TRƯỚC vạch đi bộ crosswalk)
        - STATE_STOPPED_STOP_SIGN : Dừng biển STOP (Bắt buộc dừng TRƯỚC vạch đi bộ crosswalk trong 3s)
    """
    STATE_DRIVING = "DRIVING"
    STATE_STOPPED_RED_LIGHT = "STOPPED_RED_LIGHT"
    STATE_STOPPED_STOP_SIGN = "STOPPED_STOP_SIGN"

    def __init__(self, stop_duration=3.0):
        self.state = self.STATE_DRIVING
        self.stop_duration = stop_duration
        self.stop_start_time = None
        self.active_route_command = 'STRAIGHT'
        self.detected_crosswalk = False
        self.status_message = "Driving normally"

    def set_route_command(self, cmd):
        if cmd in ROUTE_COMMANDS:
            self.active_route_command = cmd

    def update(self, detections):
        """
        Evaluates detected objects (red_light, green_light, turn_left_sign, turn_right_sign, stop_sign, crosswalk)
        and updates vehicle state & route command.
        
        Returns:
            - is_stopped (bool)       : True if vehicle must stop (throttle = 0.0)
            - route_command (str)     : Active route command ('LEFT', 'STRAIGHT', 'RIGHT')
            - status_msg (str)        : Status description for UI display
        """
        now = time.time()
        
        # 1. Parse detected object classes
        has_red_light = any(d['class_name'] == 'red_light' for d in detections)
        has_green_light = any(d['class_name'] == 'green_light' for d in detections)
        has_stop_sign = any(d['class_name'] == 'stop_sign' for d in detections)
        has_turn_left = any(d['class_name'] == 'turn_left_sign' for d in detections)
        has_turn_right = any(d['class_name'] == 'turn_right_sign' for d in detections)
        
        # Check pedestrian crosswalk line and calculate proximity (y2 in range 0..1)
        crosswalk_dets = [d for d in detections if d['class_name'] == 'crosswalk']
        self.detected_crosswalk = len(crosswalk_dets) > 0
        
        # Crosswalk is considered "CLOSE IN FRONT OF CAR" when bottom y2 >= 0.60
        crosswalk_is_close = False
        if self.detected_crosswalk:
            max_y2 = max([d['bbox'][3] for d in crosswalk_dets]) # bbox: [x1, y1, x2, y2]
            if max_y2 >= 0.60:
                crosswalk_is_close = True

        # Auto-update Route Command from traffic signs if detected
        if has_turn_left:
            self.active_route_command = 'LEFT'
        elif has_turn_right:
            self.active_route_command = 'RIGHT'

        # 2. Advanced Combined State Machine Transitions
        # Logic: Crosswalk alone (Green light / No STOP sign) -> PASS NORMALLY!
        # STOP ONLY WHEN: (Red Light or STOP Sign) AND (Crosswalk is close in front of car or close proximity)
        if self.state == self.STATE_DRIVING:
            if has_red_light and (crosswalk_is_close or not self.detected_crosswalk):
                self.state = self.STATE_STOPPED_RED_LIGHT
                self.status_message = "RED LIGHT + CROSSWALK CLOSE -> Stopped RIGHT BEFORE Pedestrian Crosswalk!"
            elif has_stop_sign and (crosswalk_is_close or not self.detected_crosswalk):
                self.state = self.STATE_STOPPED_STOP_SIGN
                self.stop_start_time = now
                self.status_message = "STOP SIGN + CROSSWALK CLOSE -> Stopped RIGHT BEFORE Pedestrian Crosswalk (3s)!"
            elif has_green_light or (not has_red_light and not has_stop_sign):
                self.status_message = "Green Light / Clear -> Passing Crosswalk Normally"

        elif self.state == self.STATE_STOPPED_RED_LIGHT:
            if has_green_light or not has_red_light:
                self.state = self.STATE_DRIVING
                self.status_message = "GREEN LIGHT -> Resuming Driving through Crosswalk!"

        elif self.state == self.STATE_STOPPED_STOP_SIGN:
            if now - self.stop_start_time >= self.stop_duration:
                self.state = self.STATE_DRIVING
                self.status_message = "STOP Sign 3s Wait Completed -> Resuming Driving through Crosswalk!"

        is_stopped = (self.state != self.STATE_DRIVING)
        return is_stopped, self.active_route_command, self.status_message

