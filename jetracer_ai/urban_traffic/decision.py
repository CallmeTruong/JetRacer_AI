import time


class IntersectionDecisionMaker:
    """
    Decision Making Layer for Intersection Traversal
    Integrates:
      - YOLO Traffic Sign Command
      - Road Geometry Model (Physical lane availability: STRAIGHT, LEFT, RIGHT)
      - Bounding Box Area and Y-coordinate proximity
    Bo ton 100% logic x l giao l, Cooldown timer v Memory hold buffer t jetracer-car.
    """

    def __init__(
        self,
        controller,
        min_area_trigger=3500,  # Bounding box area threshold for turn activation
        y_bottom_trigger=320,   # Y-max screen coordinate threshold for proximity trigger
        cooldown_time=3.0,      # Cooldown lock-out duration after completing turn (seconds)
        memory_hold_time=0.8,   # Memory buffer hold duration for missed YOLO frames (seconds)
    ):
        self.controller = controller
        self.min_area_trigger = min_area_trigger
        self.y_bottom_trigger = y_bottom_trigger
        self.cooldown_time = cooldown_time
        self.memory_hold_time = memory_hold_time

        self.last_action_time = 0
        self.last_detected_sign = None
        self.last_detected_time = 0

    def process_detections(self, detections, lane_steering, possible_directions):
        """
        detections: Detections dictionary list from YOLO [{'label': 'TURN_LEFT', 'box': [...]}]
        lane_steering: Steering angle computed from ONNX road following model.
        possible_directions: List/Dict of available lane paths.
                             Example: ['STRAIGHT', 'LEFT'] or {'straight': True, 'left': True, 'right': False}
        """
        now = time.time()

        # Normalize available paths to uppercase strings
        if isinstance(possible_directions, dict):
            available_paths = [k.upper() for k, v in possible_directions.items() if v]
        else:
            available_paths = [d.upper() for d in possible_directions]

        # 1. COOLDOWN State: Recently executed turn, temporary lockout to prevent duplicate turns
        if now - self.last_action_time < self.cooldown_time:
            self.controller.set_steering(lane_steering)
            self.controller.set_throttle(self.controller.base_throttle)
            return "STATE_COOLDOWN"

        # 2. Identify target traffic sign in current frame
        target_sign = None
        for det in detections:
            lbl = det.get('label', det.get('class_name', ''))
            if lbl in ['TURN_LEFT', 'TURN_RIGHT', 'STOP', 'RED_LIGHT', 'left-turn-sign', 'right-turn-sign', 'red-light', 'prohibition-sign']:
                target_sign = det
                break

        # 3. Memory Buffer to handle temporary YOLO frame drops
        if target_sign is not None:
            self.last_detected_sign = target_sign
            self.last_detected_time = now
        else:
            if self.last_detected_sign is not None and (now - self.last_detected_time < self.memory_hold_time):
                target_sign = self.last_detected_sign

        # 4. If no target sign detected -> Fall back to standard lane following
        if target_sign is None:
            self.controller.set_steering(lane_steering)
            self.controller.set_throttle(self.controller.base_throttle)
            return f"STATE_LANE_FOLLOWING (Paths: {available_paths})"

        # 5. Extract bounding box dimensions and metrics
        box = target_sign.get('box', target_sign.get('bbox', [0, 0, 0, 0]))
        if len(box) == 4 and box[2] > box[0]:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            y_max = y2
        else:
            x1, y1, w, h = box
            box_area = w * h
            y_max = y1 + h

        label = target_sign.get('label', target_sign.get('class_name', ''))
        # Normalize label names
        if label in ['red-light', 'prohibition-sign', 'RED_LIGHT', 'STOP']:
            norm_label = 'STOP'
        elif label in ['left-turn-sign', 'TURN_LEFT']:
            norm_label = 'TURN_LEFT'
        elif label in ['right-turn-sign', 'TURN_RIGHT']:
            norm_label = 'TURN_RIGHT'
        else:
            norm_label = label

        # 6. Handle RED LIGHT / STOP signs
        if norm_label == 'STOP':
            if box_area > 1500:
                self.controller.stop()
                return f"STATE_WAITING_{norm_label}"

        # 7. Handle LEFT TURN / RIGHT TURN at intersections
        if norm_label in ['TURN_LEFT', 'TURN_RIGHT']:
            is_close_enough = (box_area >= self.min_area_trigger) or (y_max >= self.y_bottom_trigger)

            if not is_close_enough:
                # Approach intersection at reduced speed
                self.controller.set_steering(lane_steering)
                self.controller.set_throttle(self.controller.base_throttle * 0.8)
                return f"STATE_APPROACHING_{norm_label} (Area: {int(box_area)})"
            else:
                # ACTIVATION ZONE REACHED -> Cross-reference with Road Geometry Model
                if norm_label == 'TURN_LEFT':
                    if 'LEFT' in available_paths or 'TURN_LEFT' in available_paths or not available_paths:
                        print(f"[ACTION CONFIRMED] LEFT TURN sign + Valid LEFT lane path confirmed!")
                        self.controller.move_cm(distance_cm=15, speed_factor=1.0)
                        self.controller.turn_left(duration=1.2)

                        self.last_action_time = time.time()
                        self.last_detected_sign = None
                        return "STATE_EXECUTED_TURN_LEFT"
                    else:
                        print(f"[WARNING] LEFT TURN sign detected but NO valid left lane! Cancelling turn.")
                        self.controller.set_steering(lane_steering)
                        self.controller.set_throttle(self.controller.base_throttle)
                        return "STATE_CANCELLED_NO_LEFT_LANE"

                elif norm_label == 'TURN_RIGHT':
                    if 'RIGHT' in available_paths or 'TURN_RIGHT' in available_paths or not available_paths:
                        print(f"[ACTION CONFIRMED] RIGHT TURN sign + Valid RIGHT lane path confirmed!")
                        self.controller.move_cm(distance_cm=15, speed_factor=1.0)
                        self.controller.turn_right(duration=1.0)

                        self.last_action_time = time.time()
                        self.last_detected_sign = None
                        return "STATE_EXECUTED_TURN_RIGHT"
                    else:
                        print(f"[WARNING] RIGHT TURN sign detected but NO valid right lane! Cancelling turn.")
                        self.controller.set_steering(lane_steering)
                        self.controller.set_throttle(self.controller.base_throttle)
                        return "STATE_CANCELLED_NO_RIGHT_LANE"

        return "STATE_LANE_FOLLOWING"
