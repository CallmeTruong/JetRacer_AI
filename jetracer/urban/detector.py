import os
import cv2
import numpy as np
from jetracer.urban.config import DETECTION_CLASSES

class UrbanObjectDetector:
    """
    ONNX Runtime / TensorRT Inference Wrapper for Object Detection.
    Detects 6 Urban Classes:
      0: green_light
      1: red_light
      2: turn_left_sign
      3: turn_right_sign
      4: stop_sign
      5: crosswalk (xe phải dừng trước vạch đi bộ này khi có đèn đỏ/biển STOP)
    """
    def __init__(self, model_path=None, conf_threshold=0.45, providers=None):
        self.conf_threshold = conf_threshold
        self.classes = DETECTION_CLASSES
        self.session = None
        
        if model_path and os.path.exists(model_path):
            import onnxruntime as ort
            if providers is None:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
                print(f"[+] Urban Detector ONNX Session loaded successfully with providers: {self.session.get_providers()}")
            except Exception as e:
                print(f"[!] Urban Detector ONNX Load Exception ({e}). Falling back to CPU.")
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def detect(self, cv_image):
        """
        Runs object detection on OpenCV BGR image (224x224).
        Returns a list of dicts:
            [
              {
                'class_name': 'red_light',
                'class_id': 1,
                'confidence': 0.92,
                'bbox': [x_min, y_min, x_max, y_max], # normalized 0..1 or pixel coords
                'area': width * height
              },
              ...
            ]
        """
        if self.session is None:
            # Mock / Dummy detection fallback if detector model not yet trained
            return []
            
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        # Preprocess OpenCV image
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_chw = img_norm.transpose(2, 0, 1)
        input_tensor = np.expand_dims(img_chw, axis=0)
        
        outputs = self.session.run([output_name], {input_name: input_tensor})
        detections = self._postprocess(outputs[0], cv_image.shape[1], cv_image.shape[0])
        return detections

    def _postprocess(self, raw_output, img_w, img_h):
        detections = []
        # Support standard SSD/YOLO format: (num_dets, 6) -> [x1, y1, x2, y2, score, class_id]
        if len(raw_output.shape) == 3:
            raw_output = raw_output[0]
            
        for det in raw_output:
            if len(det) < 6:
                continue
            score = float(det[4])
            class_id = int(det[5])
            if score >= self.conf_threshold and 0 <= class_id < len(self.classes):
                x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                detections.append({
                    'class_name': self.classes[class_id],
                    'class_id': class_id,
                    'confidence': score,
                    'bbox': [x1, y1, x2, y2],
                    'area': w * h
                })
        return detections
