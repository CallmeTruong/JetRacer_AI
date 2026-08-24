import time
try:
    from jetracer.urban.config import ROUTE_COMMANDS
except ImportError:
    from .config import ROUTE_COMMANDS


class UrbanFSMPlanner:
    """
    Finite State Machine (FSM) High-Level Decision Planner for Urban Autonomous Driving.
    Integrates TrafficFSM logic from smart_city.ipynb.

    Handles:
      - Red Light / red-light: STOP before pedestrian crosswalk
      - STOP Sign / prohibition-sign: STOP before pedestrian crosswalk for stop_duration (3s)
      - Turn Left / Right Sign / left-turn-sign / right-turn-sign: Auto-update route command
      - Pedestrian Crosswalk: Proximity boundary check (max_y2 >= 0.55)
    """
    STATE_DRIVING           = "DRIVING"
    STATE_STOPPED_RED_LIGHT = "STOPPED_RED_LIGHT"
    STATE_STOPPED_STOP_SIGN = "STOPPED_STOP_SIGN"

    def __init__(self, stop_duration=3.0, min_bbox_area=300, roi_x_min=0.05, roi_x_max=0.95):
        self.state = self.STATE_DRIVING
        self.stop_duration = stop_duration
        self.stop_start_time = None
        self.active_route_command = 'STRAIGHT'
        self.min_bbox_area = min_bbox_area
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max
        self.status_message = "DRIVING"

    def set_route_command(self, cmd):
        if cmd in ROUTE_COMMANDS:
            self.active_route_command = cmd

    def _normalize_class_name(self, name):
        n = str(name).lower().replace('-', '_')
        if n in ['red_light', 'red', 'stop_light']:
            return 'red_light'
        if n in ['green_light', 'green', 'go_light']:
            return 'green_light'
        if n in ['turn_left_sign', 'left_turn_sign', 'left_turn', 'turn_left']:
            return 'turn_left_sign'
        if n in ['turn_right_sign', 'right_turn_sign', 'right_turn', 'turn_right']:
            return 'turn_right_sign'
        if n in ['stop_sign', 'prohibition_sign', 'stop']:
            return 'stop_sign'
        if n in ['crosswalk', 'pedestrian', 'zebra']:
            return 'crosswalk'
        return n

    def update(self, detections):
        """
        Evaluates detected objects (red_light, green_light, turn_left_sign, turn_right_sign, stop_sign, crosswalk).

        Returns:
            - is_stopped (bool)   : True if vehicle must stop (throttle = 0.0)
            - route_command (str) : Active route command ('LEFT', 'STRAIGHT', 'RIGHT')
            - status_msg (str)    : Status description for UI display
        """
        now = time.time()
        if not detections:
            if self.state == self.STATE_STOPPED_STOP_SIGN:
                if now - self.stop_start_time >= self.stop_duration:
                    self.state = self.STATE_DRIVING
                    self.status_message = "DRIVING (STOP 3s done)"
            is_stopped = (self.state != self.STATE_DRIVING)
            return is_stopped, self.active_route_command, self.status_message

        # Filter valid spatial detections
        valid_dets = []
        for d in detections:
            c_name = self._normalize_class_name(d['class_name'])
            bbox = d.get('bbox', [0, 0, 1, 1])
            # Check ROI center X ratio (0.05..0.95)
            cx = (bbox[0] + bbox[2]) / 2.0
            if self.roi_x_min <= cx <= self.roi_x_max:
                d_copy = dict(d)
                d_copy['norm_class'] = c_name
                valid_dets.append(d_copy)

        has_red_light   = any(d['norm_class'] == 'red_light' for d in valid_dets)
        has_green_light = any(d['norm_class'] == 'green_light' for d in valid_dets)
        has_stop_sign   = any(d['norm_class'] == 'stop_sign' for d in valid_dets)
        has_turn_left   = any(d['norm_class'] == 'turn_left_sign' for d in valid_dets)
        has_turn_right  = any(d['norm_class'] == 'turn_right_sign' for d in valid_dets)

        crosswalk_dets     = [d for d in valid_dets if d['norm_class'] == 'crosswalk']
        detected_crosswalk = len(crosswalk_dets) > 0
        crosswalk_is_close = False
        if detected_crosswalk:
            max_y2 = max([d['bbox'][3] for d in crosswalk_dets])
            if max_y2 >= 0.55:  # crosswalk line in front of car
                crosswalk_is_close = True

        # Auto-update Route Command from detected turn signs
        if has_turn_left:
            self.active_route_command = 'LEFT'
        elif has_turn_right:
            self.active_route_command = 'RIGHT'

        # FSM State Transitions
        if self.state == self.STATE_DRIVING:
            if has_red_light and (crosswalk_is_close or not detected_crosswalk):
                self.state = self.STATE_STOPPED_RED_LIGHT
                self.status_message = "STOP (Red Light)"
            elif has_stop_sign and (crosswalk_is_close or not detected_crosswalk):
                self.state = self.STATE_STOPPED_STOP_SIGN
                self.stop_start_time = now
                self.status_message = "STOP (STOP Sign 3s)"
            elif has_green_light:
                self.status_message = "DRIVING (Green Light)"

        elif self.state == self.STATE_STOPPED_RED_LIGHT:
            if has_green_light or not has_red_light:
                self.state = self.STATE_DRIVING
                self.status_message = "DRIVING (Green Light)"

        elif self.state == self.STATE_STOPPED_STOP_SIGN:
            if now - self.stop_start_time >= self.stop_duration:
                self.state = self.STATE_DRIVING
                self.status_message = "DRIVING (STOP 3s done)"

        is_stopped = (self.state != self.STATE_DRIVING)
        return is_stopped, self.active_route_command, self.status_message
