"""
Standalone Python GUI Script for Multi-Modal Waypoint Trajectory Labeling.
Run this script on PC/Laptop or Jetson:
    python -m jetracer.urban.label_trajectory_gui --dataset_dir urban_dataset_A

Mouse & Keyboard Controls:
  - Left Mouse Click  : Click 5 sequential waypoints on the image (W1 -> W2 -> W3 -> W4 -> W5)
  - Key '1'           : Set Route Command to 'LEFT'
  - Key '2'           : Set Route Command to 'STRAIGHT'
  - Key '3'           : Set Route Command to 'RIGHT'
  - Key 's'           : Save Trajectory JSON file
  - Key 'r'           : Reset/Clear points for current image
  - Key 'n' / Right   : Next Image
  - Key 'p' / Left    : Previous Image
  - Key 'q' / Esc     : Quit GUI
"""

import os
import sys
import glob
import json
import cv2
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from jetracer.urban.config import ROUTE_COMMANDS, NUM_WAYPOINTS

class TrajectoryLabelerGUI:
    def __init__(self, dataset_dir):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.image_files = sorted(glob.glob(os.path.join(self.dataset_dir, "*.jpg")))
        if not self.image_files:
            print(f"[!] ERROR: No .jpg image files found in '{self.dataset_dir}'!")
            sys.exit(1)
            
        self.current_idx = 0
        self.active_cmd = 'STRAIGHT'
        self.clicked_points = []
        self.window_name = "JetRacer Trajectory Waypoint Labeler GUI"
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_points) < NUM_WAYPOINTS:
                x = int(max(0, min(224, x)))
                y = int(max(0, min(224, y)))
                self.clicked_points.append((x, y))
                print(f"  [+] Clicked Waypoint #{len(self.clicked_points)}: ({x}, {y})")
                self._draw()

    def _load_existing_annotation(self, img_path, cmd):
        base = os.path.splitext(img_path)[0]
        json_path = f"{base}_{cmd}.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                return [tuple(pt) for pt in data.get('waypoints', [])]
            except Exception:
                pass
        return []

    def _draw(self):
        img_path = self.image_files[self.current_idx]
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            return
            
        if raw_img.shape[0] != 224 or raw_img.shape[1] != 224:
            raw_img = cv2.resize(raw_img, (224, 224))
            
        display_img = raw_img.copy()
        
        # Draw line spline and waypoint circles
        pts = self.clicked_points
        for i in range(len(pts) - 1):
            cv2.line(display_img, pts[i], pts[i+1], (0, 255, 255), 2)
        for i, pt in enumerate(pts):
            cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_img, str(i+1), (pt[0]+6, pt[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
        # Draw status banner
        banner = f"[{self.current_idx+1}/{len(self.image_files)}] Cmd: {self.active_cmd} | WPs: {len(pts)}/{NUM_WAYPOINTS}"
        cv2.rectangle(display_img, (0, 0), (224, 22), (30, 30, 30), -1)
        cv2.putText(display_img, banner, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        
        cv2.imshow(self.window_name, display_img)

    def save_current(self):
        if len(self.clicked_points) < NUM_WAYPOINTS:
            print(f"[!] Warning: Need exactly {NUM_WAYPOINTS} waypoints (currently {len(self.clicked_points)}). Click on image first!")
            return
            
        img_path = self.image_files[self.current_idx]
        base = os.path.splitext(img_path)[0]
        json_path = f"{base}_{self.active_cmd}.json"
        
        data = {
            "image_path": os.path.basename(img_path),
            "route_command": self.active_cmd,
            "waypoints": [list(pt) for pt in self.clicked_points]
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"[+] Saved Annotation -> '{json_path}'")

    def run(self):
        print("\n" + "="*60)
        print(" JetRacer Multi-Modal Trajectory Waypoint Labeler GUI")
        print("="*60)
        print(" Controls:")
        print("  - Left Mouse Click : Click 5 Waypoints (W1 -> W2 -> W3 -> W4 -> W5)")
        print("  - Keys 1 / 2 / 3   : Switch Command (1: LEFT | 2: STRAIGHT | 3: RIGHT)")
        print("  - Key 's'          : Save JSON Annotation")
        print("  - Key 'r'          : Reset Clicked Points")
        print("  - Key 'n' / Right  : Next Image")
        print("  - Key 'p' / Left   : Previous Image")
        print("  - Key 'q' / Esc    : Quit")
        print("="*60 + "\n")
        
        while True:
            # Check existing annotation for current image & active command
            img_path = self.image_files[self.current_idx]
            existing = self._load_existing_annotation(img_path, self.active_cmd)
            if existing and not self.clicked_points:
                self.clicked_points = existing
                
            self._draw()
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                self.active_cmd = 'LEFT'
                self.clicked_points = self._load_existing_annotation(img_path, 'LEFT')
                print("[*] Route Command set to: LEFT")
            elif key == ord('2'):
                self.active_cmd = 'STRAIGHT'
                self.clicked_points = self._load_existing_annotation(img_path, 'STRAIGHT')
                print("[*] Route Command set to: STRAIGHT")
            elif key == ord('3'):
                self.active_cmd = 'RIGHT'
                self.clicked_points = self._load_existing_annotation(img_path, 'RIGHT')
                print("[*] Route Command set to: RIGHT")
            elif key == ord('s'):
                self.save_current()
            elif key == ord('r'):
                self.clicked_points = []
                print("[*] Reset waypoints for current image.")
            elif key == ord('n') or key == 83: # Next image
                self.save_current()
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                    self.clicked_points = []
            elif key == ord('p') or key == 81: # Previous image
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.clicked_points = []
                    
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Modal Waypoint Trajectory Labeler GUI")
    parser.add_argument('--dataset_dir', type=str, default='urban_dataset_A', help='Path to directory containing .jpg images')
    args = parser.parse_args()
    
    labeler = TrajectoryLabelerGUI(args.dataset_dir)
    labeler.run()
