import time


class TrafficFSM:
    """
    Traffic Sign Finite State Machine (TrafficFSM).
    Implements spatial filtering (BBox area & ROI ratio) and priority-based traffic sign resolution.
    """

    def __init__(
        self,
        default_state='FORWARD',
        conf_threshold=0.5,
        min_consecutive_frames=3,
        state_timeout=2.0,
        min_bbox_area=900,   # Minimum bounding box area threshold to filter out distant or adjacent lane signs
        roi_x_min=0.05,     # Exclude 5% left screen margin
        roi_x_max=0.95,     # Exclude 5% right screen margin
    ):
        self.STATE_STOP = 'STOP'
        self.STATE_FORWARD = 'FORWARD'
        self.STATE_TURN_LEFT = 'TURN_LEFT'
        self.STATE_TURN_RIGHT = 'TURN_RIGHT'

        self.default_state = default_state
        self.conf_threshold = conf_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.state_timeout = state_timeout

        # (translated)
        self.min_bbox_area = min_bbox_area
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max

        # Tracking state
        self.current_state = self.default_state
        self.last_detected_label = None
        self.consecutive_count = 0
        self.last_detection_time = time.time()

        self.class_to_state = {
            'red-light': self.STATE_STOP,
            'prohibition-sign': self.STATE_STOP,
            'green-light': self.STATE_FORWARD,
            'straight-ahead-sign': self.STATE_FORWARD,
            'left-turn-sign': self.STATE_TURN_LEFT,
            'right-turn-sign': self.STATE_TURN_RIGHT,
            # Supporting alternate class names
            'RED_LIGHT': self.STATE_STOP,
            'STOP': self.STATE_STOP,
            'GREEN_LIGHT': self.STATE_FORWARD,
            'TURN_LEFT': self.STATE_TURN_LEFT,
            'TURN_RIGHT': self.STATE_TURN_RIGHT,
        }

        self.priority = {
            'red-light': 4,
            'prohibition-sign': 4,
            'RED_LIGHT': 4,
            'STOP': 4,
            'green-light': 3,
            'GREEN_LIGHT': 3,
            'left-turn-sign': 2,
            'right-turn-sign': 2,
            'TURN_LEFT': 2,
            'TURN_RIGHT': 2,
            'straight-ahead-sign': 1,
        }

    def _is_valid_spatial_detection(self, det, img_w, img_h):
        """Checks whether detected sign resides within the vehicle's spatial driving ROI."""
        box = det.get('bbox', det.get('box', None))
        if box is None:
            return True

        x1, y1, x2, y2 = box[0], box[1], box[2] if len(box) == 4 and box[2] > box[0] else box[0] + box[2], box[3] if len(box) == 4 and box[3] > box[1] else box[1] + box[3]
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        area = w * h

        # 1. Filter by area (too small = distant or adjacent lane sign)
        if area < self.min_bbox_area:
            return False

        # 2. Filter by ROI (sign located too close to screen margins)
        center_x_ratio = ((x1 + x2) / 2.0) / max(img_w, 1.0)
        if not (self.roi_x_min <= center_x_ratio <= self.roi_x_max):
            return False

        return True

    def _select_best_detection(self, detections, img_w, img_h):
        """Filters valid detections considering confidence thresholds and spatial location."""
        valid_dets = []
        for d in detections:
            conf = d.get('confidence', d.get('score', 0.0))
            c_name = d.get('class_name', d.get('label', ''))
            if conf >= self.conf_threshold and c_name in self.class_to_state:
                if self._is_valid_spatial_detection(d, img_w, img_h):
                    valid_dets.append(d)

        if not valid_dets:
            return None

        def get_area(d):
            box = d.get('bbox', d.get('box', [0, 0, 0, 0]))
            w = abs(box[2] - box[0]) if len(box) == 4 and box[2] > box[0] else box[2]
            h = abs(box[3] - box[1]) if len(box) == 4 and box[3] > box[1] else box[3]
            return w * h

        # Priority sorting: 1. Sign Priority -> 2. Largest BBox Area -> 3. Confidence score
        valid_dets.sort(
            key=lambda d: (
                self.priority.get(d.get('class_name', d.get('label', '')), 0),
                get_area(d),
                d.get('confidence', d.get('score', 0.0)),
            ),
            reverse=True,
        )
        return valid_dets[0].get('class_name', valid_dets[0].get('label', ''))

    def update(self, detections, img_w=640, img_h=480):
        """Updates FSM state based on spatial detections."""
        now = time.time()
        best_label = self._select_best_detection(detections, img_w, img_h)

        if best_label is not None:
            if best_label == self.last_detected_label:
                self.consecutive_count += 1
            else:
                self.last_detected_label = best_label
                self.consecutive_count = 1

            if self.consecutive_count >= self.min_consecutive_frames:
                self.current_state = self.class_to_state[best_label]
                self.last_detection_time = now
        else:
            if now - self.last_detection_time > self.state_timeout:
                self.current_state = self.default_state
                self.last_detected_label = None
                self.consecutive_count = 0

        return self.current_state


# Keep alias UrbanFSMPlanner compatible with current jetracer_ai imports
UrbanFSMPlanner = TrafficFSM
