import cv2
import numpy as np


class YOLOProcessor:
    """
    Preprocessing, Postprocessing, and NMS for YOLO Detection.
    NumPy NMS algorithm and input resolution normalization.
    """

    def __init__(
        self, img_size=640, conf_thresh=0.45, iou_thresh=0.45, classes=None
    ):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes or [
            'green-light',
            'left-turn-sign',
            'prohibition-sign',
            'red-light',
            'right-turn-sign',
            'straight-ahead-sign',
        ]

    def _nms_numpy(self, boxes, scores):
        """Pure NumPy Non-Maximum Suppression (NMS) implementation."""
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)  # [x, y, w, h]
        scores = np.array(scores)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def preprocess(self, img0):
        h, w = img0.shape[:2]
        img = cv2.resize(img0, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(img), h, w

    def postprocess(self, output, original_h, original_w):
        predictions = np.squeeze(output).T
        boxes, confidences, class_ids = [], [], []

        x_factor = original_w / self.img_size
        y_factor = original_h / self.img_size

        for row in predictions:
            classes_scores = row[4:]
            class_id = int(np.argmax(classes_scores))
            confidence = float(classes_scores[class_id])

            if confidence >= self.conf_thresh:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                boxes.append([left, top, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = self._nms_numpy(boxes, confidences)

        results = []
        for i in indices:
            results.append({
                'box': boxes[i],
                'bbox': [boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]],
                'confidence': confidences[i],
                'class_name': self.classes[class_ids[i]],
                'label': self.classes[class_ids[i]],
            })
        return results

    def draw_bboxes(self, img0, detections):
        for det in detections:
            x, y, w, h = det['box']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(img0, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img0,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return img0


class RoadProcessor:
    """
    Road Obstacle Safety Processor (Classification: FREE vs BLOCKED).
    ImageNet normalization and Sigmoid/Softmax logit postprocessing.
    """

    def __init__(self, img_size=(160, 160), threshold=0.5):
        self.img_size = img_size
        self.threshold = threshold

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame):
        img = cv2.resize(frame, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, raw_output):
        logits = np.array(raw_output).squeeze()

        # Sigmoid output
        if logits.ndim == 0 or logits.size == 1:
            logit_val = float(logits)
            prob_blocked = 1.0 / (1.0 + np.exp(-logit_val))
        # Softmax output [logit_free, logit_blocked]
        else:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            prob_blocked = float(probs[1])

        status = 'BLOCKED' if prob_blocked >= self.threshold else 'FREE'

        return {
            'status': status,
            'blocked_probability': round(prob_blocked, 4),
            'free_probability': round(1.0 - prob_blocked, 4),
        }
