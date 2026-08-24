import os
import cv2
import numpy as np
from jetracer_ai.urban_traffic.constants import DETECTION_CLASSES



class UrbanObjectDetector:
    """
    ONNX Runtime / TensorRT Inference Wrapper for Urban Object Detection (YOLOv5 / YOLOv8 / SSD).
    Detects 6 Urban Classes:
      0: green_light
      1: red_light
      2: turn_left_sign
      3: turn_right_sign
      4: stop_sign
      5: crosswalk (pedestrian crosswalk line)

    FIXES vs original:
      1. `_postprocess` no longer assumes box coords are always in 224px pixel-space
         before dividing by 224.0. It now checks the ACTUAL max value of the box
         columns and only rescales if the values look like pixel coordinates
         (max > ~1.5). This was the most likely reason bboxes were invisible/collapsed
         to a corner: if the exported model already outputs normalized [0..1] boxes,
         the old code divided them by 224 again, shrinking every box toward (0,0).
      2. Correctly distinguishes YOLOv5-style output (has an objectness/"obj_conf"
         column: 4 box + 1 objectness + N classes) from YOLOv8-style output
         (4 box + N classes, no objectness). The old code always treated column 4
         onward as class scores, which silently drops all detections if there IS
         an objectness column (real confidence = obj_conf * class_conf, not just
         class_conf), or double-counts if raw_output.shape[1] == 6 case never
         actually matches the real model shape.
      3. One-time diagnostic print of raw output shape / value range on first call,
         so a mismatched export (e.g. input size 320/640 instead of 224) is easy to
         spot from the logs instead of silently returning zero detections.
    """
    def __init__(self, model_path=None, conf_threshold=0.35, providers=None, input_size=224):
        self.conf_threshold = conf_threshold
        self.classes         = DETECTION_CLASSES
        self.num_classes      = len(self.classes)
        self.session          = None
        self.input_size       = input_size
        self._diagnostics_printed = False

        if model_path and os.path.exists(model_path):
            import onnxruntime as ort
            if providers is None:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
                print(f"[+] Urban Object Detector ONNX Session loaded: {self.session.get_providers()}")
            except Exception as e:
                print(f"[!] Urban Detector Load Exception ({e}). Falling back to CPU.")
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def detect(self, cv_image):
        """
        Runs object detection on OpenCV BGR image (224x224).
        Returns a list of dicts:
            [
              {
                'class_name': 'red_light',
                'class_id': 1,
                'confidence': 0.88,
                'bbox': [x_min, y_min, x_max, y_max],  # normalized 0..1
                'area': w * h
              }, ...
            ]
        """
        if self.session is None:
            return []

        input_name  = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        # Preprocess OpenCV image
        img_rgb      = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_resized  = cv2.resize(img_rgb, (self.input_size, self.input_size))
        img_norm     = img_resized.astype(np.float32) / 255.0
        img_chw      = img_norm.transpose(2, 0, 1)
        input_tensor = np.expand_dims(img_chw, axis=0)

        outputs = self.session.run([output_name], {input_name: input_tensor})

        if not self._diagnostics_printed:
            raw = outputs[0]
            print(f"[detector diag] raw output shape={raw.shape}, "
                  f"min={float(raw.min()):.3f}, max={float(raw.max()):.3f}")
            self._diagnostics_printed = True

        detections = self._postprocess(outputs[0], cv_image.shape[1], cv_image.shape[0])
        return detections

    def _postprocess(self, raw_output, img_w, img_h):
        detections = []
        if raw_output is None or raw_output.size == 0:
            return detections

        # Squeeze batch dimension if (1, C, N) or (1, N, C)
        if raw_output.ndim == 3:
            raw_output = raw_output[0]

        num_classes = self.num_classes  # 6

        # ── Normalize orientation to (num_anchors, num_attrs) ─────────────────
        # Ultralytics YOLOv8/v5 export sometimes comes out as (attrs, anchors)
        # e.g. (10, 8400) or (11, 8400) -- transpose so rows = detections.
        if raw_output.shape[0] < raw_output.shape[1] and raw_output.shape[0] in (
            4 + num_classes,       # YOLOv8 style: 4 box + N classes, no objectness
            5 + num_classes,       # YOLOv5 style: 4 box + 1 objectness + N classes
            6,                    # pre-NMS [x1,y1,x2,y2,score,class_id]
        ):
            raw_output = raw_output.T

        n_attrs = raw_output.shape[1]

        # ── Case A: Pre-NMS format (num_dets, 6) -> [x1, y1, x2, y2, score, class_id] ──
        if n_attrs == 6:
            for det in raw_output:
                score    = float(det[4])
                class_id = int(det[5])
                if score >= self.conf_threshold and 0 <= class_id < num_classes:
                    x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                    x1, y1, x2, y2 = self._normalize_box(x1, y1, x2, y2, raw_output[:, :4])
                    detections.append({
                        'class_name': self.classes[class_id],
                        'class_id': class_id,
                        'confidence': score,
                        'bbox': [x1, y1, x2, y2],
                        'area': abs(x2 - x1) * abs(y2 - y1)
                    })
            return detections

        # ── Case B: Raw grid output, box + (optional objectness) + class scores ──
        boxes = raw_output[:, :4]

        if n_attrs == 5 + num_classes:
            # YOLOv5-style: column 4 = objectness, columns 5: = per-class scores
            objectness   = raw_output[:, 4]
            class_scores = raw_output[:, 5:]
            combined     = objectness[:, None] * class_scores
        elif n_attrs == 4 + num_classes:
            # YOLOv8-style: no separate objectness, columns 4: = per-class scores
            combined = raw_output[:, 4:]
        elif n_attrs > 4:
            # Unknown layout with extra columns -- fall back to treating everything
            # after the box as class scores (best effort, matches old behavior).
            combined = raw_output[:, 4:4 + num_classes] if raw_output.shape[1] >= 4 + num_classes else raw_output[:, 4:]
        else:
            return detections

        max_scores  = np.max(combined, axis=1)
        class_ids   = np.argmax(combined, axis=1)
        keep_mask   = max_scores >= self.conf_threshold

        if not np.any(keep_mask):
            return detections

        kept_boxes  = boxes[keep_mask]
        kept_scores = max_scores[keep_mask]
        kept_ids    = class_ids[keep_mask]

        # Decide once, from the FULL box coordinate range, whether these are already
        # normalized [0..1] or in pixel/grid space -- do NOT blindly divide by 224.
        all_x = np.concatenate([boxes[:, 0], boxes[:, 2]])
        all_y = np.concatenate([boxes[:, 1], boxes[:, 3]])
        looks_normalized = (np.max(np.abs(all_x)) <= 1.5) and (np.max(np.abs(all_y)) <= 1.5)

        for box, score, class_id in zip(kept_boxes, kept_scores, kept_ids):
            if not (0 <= class_id < num_classes):
                continue
            cx, cy, bw, bh = box
            if looks_normalized:
                x1 = float(cx - bw / 2.0)
                y1 = float(cy - bh / 2.0)
                x2 = float(cx + bw / 2.0)
                y2 = float(cy + bh / 2.0)
            else:
                x1 = float((cx - bw / 2.0) / self.input_size)
                y1 = float((cy - bh / 2.0) / self.input_size)
                x2 = float((cx + bw / 2.0) / self.input_size)
                y2 = float((cy + bh / 2.0) / self.input_size)

            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)

            detections.append({
                'class_name': self.classes[class_id],
                'class_id': int(class_id),
                'confidence': float(score),
                'bbox': [x1, y1, x2, y2],
                'area': abs(x2 - x1) * abs(y2 - y1)
            })

        # Apply Non-Maximum Suppression (NMS) to eliminate duplicate boxes
        if len(detections) > 1:
            detections = self._nms(detections, iou_threshold=0.45)

        return detections

    def _normalize_box(self, x1, y1, x2, y2, all_boxes):
        """Normalize a single box to [0..1], auto-detecting pixel vs normalized space
        from the full batch of boxes rather than assuming a fixed 224px frame."""
        all_x = np.concatenate([all_boxes[:, 0], all_boxes[:, 2]])
        all_y = np.concatenate([all_boxes[:, 1], all_boxes[:, 3]])
        looks_normalized = (np.max(np.abs(all_x)) <= 1.5) and (np.max(np.abs(all_y)) <= 1.5)
        if looks_normalized:
            return x1, y1, x2, y2
        return x1 / self.input_size, y1 / self.input_size, x2 / self.input_size, y2 / self.input_size

    def _nms(self, dets, iou_threshold=0.45):
        if not dets:
            return []
        dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [
                d for d in dets
                if d['class_name'] != best['class_name'] or self._iou(best['bbox'], d['bbox']) < iou_threshold
            ]
        return keep

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea  = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea  = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)