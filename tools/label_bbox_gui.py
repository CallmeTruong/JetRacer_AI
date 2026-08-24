"""
Standalone Python GUI Script for Object Detection Bounding Box Labeling.
Run this script on PC/Laptop or Jetson:
    python -m jetracer.urban.label_bbox_gui --dataset_dir urban_dataset_A

Mouse & Keyboard Controls:
  - Mouse Drag (Left Click & Drag) : Draw bounding box rectangle over object
  - Keys '1' .. '6'                 : Select Active Class:
                                       1: green_light
                                       2: red_light
                                       3: turn_left_sign
                                       4: turn_right_sign
                                       5: stop_sign
                                       6: crosswalk
  - Key 's'                         : Save YOLO format .txt annotation file
  - Key 'c'                         : Clear bounding boxes for current image
  - Key 'n' / Right                 : Next Image
  - Key 'p' / Left                  : Previous Image
  - Key 'q' / Esc                   : Quit GUI
"""

import os
import sys
import glob
import cv2
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

try:
    from jetracer.urban.config import DETECTION_CLASSES
except ImportError:
    from .config import DETECTION_CLASSES


class BBoxLabelerGUI:
    def __init__(self, dataset_dir):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.image_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.jpg")))
        if not self.image_files:
            print(f"[!] ERROR: No .jpg image files found in '{self.dataset_dir}'!")
            sys.exit(1)
            
        self.current_idx = 0
        self.active_class_idx = 0  # Default: green_light
        self.bboxes = [] # List of dicts: [{'class_id': int, 'bbox': [x1, y1, x2, y2]}]
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.curr_x, self.curr_y = -1, -1
        
        self.window_name = "JetRacer Object Detector Bounding Box Labeler GUI"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.curr_x, self.curr_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.curr_x, self.curr_y = x, y
                self._draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.ix, x), min(self.iy, y)
            x2, y2 = max(self.ix, x), max(self.iy, y)
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                self.bboxes.append({
                    'class_id': self.active_class_idx,
                    'class_name': DETECTION_CLASSES[self.active_class_idx],
                    'bbox': [x1, y1, x2, y2]
                })
                print(f"  [+] Added BBox: {DETECTION_CLASSES[self.active_class_idx]} -> [{x1}, {y1}, {x2}, {y2}]")
            self._draw()

    def _load_existing_yolo_txt(self, img_path):
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        bboxes = []
        if os.path.exists(txt_path):
            try:
                raw_img = cv2.imread(img_path)
                h, w = raw_img.shape[:2]
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = int((cx - bw / 2.0) * w)
                        y1 = int((cy - bh / 2.0) * h)
                        x2 = int((cx + bw / 2.0) * w)
                        y2 = int((cy + bh / 2.0) * h)
                        bboxes.append({
                            'class_id': cls_id,
                            'class_name': DETECTION_CLASSES[cls_id] if cls_id < len(DETECTION_CLASSES) else 'unknown',
                            'bbox': [x1, y1, x2, y2]
                        })
            except Exception:
                pass
        return bboxes

    def _draw(self):
        img_path = self.image_files[self.current_idx]
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            return
            
        if raw_img.shape[0] != 224 or raw_img.shape[1] != 224:
            raw_img = cv2.resize(raw_img, (224, 224))
            
        display_img = raw_img.copy()
        
        # Draw existing bounding boxes
        for item in self.bboxes:
            x1, y1, x2, y2 = item['bbox']
            cls_name = item['class_name']
            color = (0, 255, 0) if cls_name == 'green_light' else (0, 0, 255) if cls_name in ['red_light', 'stop_sign'] else (255, 255, 0)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_img, cls_name, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        # Draw transient dragging box
        if self.drawing:
            cv2.rectangle(display_img, (self.ix, self.iy), (self.curr_x, self.curr_y), (255, 255, 255), 1)
            
        # Status Header Banner
        active_cls = DETECTION_CLASSES[self.active_class_idx]
        banner = f"[{self.current_idx+1}/{len(self.image_files)}] Class: {self.active_class_idx+1}-{active_cls} | BBoxes: {len(self.bboxes)}"
        cv2.rectangle(display_img, (0, 0), (224, 22), (30, 30, 30), -1)
        cv2.putText(display_img, banner, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        cv2.imshow(self.window_name, display_img)

    def save_current(self):
        if not self.bboxes:
            return
        img_path = self.image_files[self.current_idx]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        
        raw_img = cv2.imread(img_path)
        h, w = 224.0, 224.0
        
        lines = []
        for item in self.bboxes:
            x1, y1, x2, y2 = item['bbox']
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{item['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
        with open(txt_path, 'w') as f:
            f.writelines(lines)
            
        print(f"[+] Saved YOLO BBox annotation -> '{txt_path}' ({len(self.bboxes)} bboxes)")

    def run(self):
        print("\n" + "="*60)
        print(" JetRacer Object Detector Bounding Box Labeler GUI")
        print("="*60)
        print(" Target Classes:")
        for idx, cls in enumerate(DETECTION_CLASSES):
            print(f"   [{idx+1}] {cls}")
        print("-" * 60)
        print(" Controls:")
        print("  - Mouse Drag       : Draw bounding box over object")
        print("  - Keys 1 .. 6      : Select active class")
        print("  - Key 's'          : Save YOLO .txt file")
        print("  - Key 'c'          : Clear bboxes for current image")
        print("  - Key 'n' / Right  : Next Image")
        print("  - Key 'p' / Left   : Previous Image")
        print("  - Key 'q' / Esc    : Quit")
        print("="*60 + "\n")
        
        while True:
            img_path = self.image_files[self.current_idx]
            if not self.bboxes:
                self.bboxes = self._load_existing_yolo_txt(img_path)
                
            self._draw()
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif ord('1') <= key <= ord('6'):
                cls_idx = key - ord('1')
                if cls_idx < len(DETECTION_CLASSES):
                    self.active_class_idx = cls_idx
                    print(f"[*] Active BBox Class set to [{cls_idx+1}]: {DETECTION_CLASSES[cls_idx]}")
            elif key == ord('s'):
                self.save_current()
            elif key == ord('c'):
                self.bboxes = []
                print("[*] Cleared bounding boxes for current image.")
            elif key == ord('n') or key == 83: # Next
                self.save_current()
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                    self.bboxes = []
            elif key == ord('p') or key == 81: # Previous
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.bboxes = []
                    
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object Detector Bounding Box Labeler GUI")
    parser.add_argument('--dataset_dir', type=str, default='urban_dataset_A', help='Path to directory containing .jpg images')
    args = parser.parse_args()
    
    labeler = BBoxLabelerGUI(args.dataset_dir)
    labeler.run()
