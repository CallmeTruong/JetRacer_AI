"""
Standalone Python GUI Script for Multi-Task Trajectory & Free/Block Safety Labeling (Standard Camera View).
Run this script on PC/Laptop or Jetson:
    python -m jetracer.urban.label_trajectory_gui --dataset_dir urban_dataset --output_dir multitask_dataset

Mouse & Keyboard Controls:
  - Left Mouse Click  : Click 5 sequential waypoints on image (W1 -> W2 -> W3 -> W4 -> W5)
  - Key 'b'           : Mark image as BLOCKED (is_blocked=1, auto-assign safe stop WPs, auto-save & next!)
  - Key 'f'           : Mark image as FREE (is_blocked=0)
  - Key '1'           : Set Route Command to 'LEFT'
  - Key '2'           : Set Route Command to 'STRAIGHT'
  - Key '3'           : Set Route Command to 'RIGHT'
  - Key 's'           : Save JSON & Image to output labeled folder
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
import shutil
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

try:
    from jetracer.urban.config import ROUTE_COMMANDS, NUM_WAYPOINTS
except ImportError:
    from .config import ROUTE_COMMANDS, NUM_WAYPOINTS



class TrajectoryLabelerGUI:
    def __init__(self, dataset_dirs, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Collect image files from all dataset directories (including move_controller datasets)
        self.image_files = []
        for d in dataset_dirs:
            abs_d = os.path.abspath(d)
            if os.path.exists(abs_d):
                files = sorted(glob.glob(os.path.join(abs_d, "*.jpg"))) + sorted(glob.glob(os.path.join(abs_d, "*.png")))
                self.image_files.extend(files)
                # Check subfolders like move_controller/dataset/free and blocked
                sub_free = sorted(glob.glob(os.path.join(abs_d, "free", "*.jpg"))) + sorted(glob.glob(os.path.join(abs_d, "free", "*.png")))
                sub_blocked = sorted(glob.glob(os.path.join(abs_d, "blocked", "*.jpg"))) + sorted(glob.glob(os.path.join(abs_d, "blocked", "*.png")))
                self.image_files.extend(sub_free + sub_blocked)

        # Remove duplicate image paths
        seen = set()
        unique_files = []
        for f in self.image_files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        self.image_files = unique_files

        if not self.image_files:
            print(f"[!] ERROR: No image files found in specified dataset directories: {dataset_dirs}!")
            sys.exit(1)

        print(f"[+] Multi-Task Labeler Loaded {len(self.image_files)} images (Standard Camera View).")
        print(f"[+] Labeled Output Directory: '{self.output_dir}'")

        self.current_idx = 0
        self.active_cmd = 'STRAIGHT'
        self.is_blocked = 0 # 0: FREE, 1: BLOCKED
        self.clicked_points = []
        self.window_name = "JetRacer Multi-Task Trajectory & Free/Block Labeler GUI"

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._check_initial_blocked_state()

    def _check_initial_blocked_state(self):
        img_path = self.image_files[self.current_idx]
        if "blocked" in img_path.lower():
            self.is_blocked = 1
        else:
            self.is_blocked = 0

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_points) < NUM_WAYPOINTS:
                x = int(max(0, min(224, x)))
                y = int(max(0, min(224, y)))
                self.clicked_points.append((x, y))
                self.is_blocked = 0 # Explicitly set to FREE if waypoints clicked
                print(f"  [+] Clicked Waypoint #{len(self.clicked_points)}: ({x}, {y})")
                self._draw()

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
        safety_status = "BLOCKED" if self.is_blocked == 1 else "FREE"
        status_color = (0, 0, 255) if self.is_blocked == 1 else (0, 255, 0)
        banner = f"[{self.current_idx+1}/{len(self.image_files)}] Cmd: {self.active_cmd} | State: {safety_status} | WPs: {len(pts)}/{NUM_WAYPOINTS}"
        cv2.rectangle(display_img, (0, 0), (224, 22), (30, 30, 30), -1)
        cv2.putText(display_img, banner, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)

        cv2.imshow(self.window_name, display_img)

    def mark_blocked_and_save(self):
        """Key 'b': Mark image as BLOCKED, auto-assign safe stop waypoints, save & next!"""
        self.is_blocked = 1
        # Safe stop waypoints (pixel coords normalized at center bottom)
        self.clicked_points = [(112, 210)] * NUM_WAYPOINTS
        print("  [*] Marked as BLOCKED -> Auto-assigned 5 Safe Stop Waypoints.")
        self.save_current()

    def save_current(self):
        if len(self.clicked_points) < NUM_WAYPOINTS and self.is_blocked == 0:
            print(f"[!] Warning: Need exactly {NUM_WAYPOINTS} waypoints (currently {len(self.clicked_points)}). Press 'b' if BLOCKED or click 5 waypoints!")
            return

        img_path = self.image_files[self.current_idx]
        file_basename = os.path.basename(img_path)
        base_name = os.path.splitext(file_basename)[0]
        curr_cmd = self.active_cmd

        # Save labeled image copy into output_dir
        dest_img_path = os.path.join(self.output_dir, file_basename)
        if not os.path.exists(dest_img_path):
            shutil.copy(img_path, dest_img_path)

        json_filename = f"{base_name}_{curr_cmd}.json"
        dest_json_path = os.path.join(self.output_dir, json_filename)

        data = {
            "image_path": file_basename,
            "route_command": curr_cmd,
            "route_command_index": ROUTE_COMMANDS.index(curr_cmd) if curr_cmd in ROUTE_COMMANDS else 1,
            "is_blocked": self.is_blocked,
            "status": "BLOCKED" if self.is_blocked == 1 else "FREE",
            "waypoints": [list(pt) for pt in self.clicked_points]
        }

        with open(dest_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[+] Saved Label JSON -> '{dest_json_path}' (State: {'BLOCKED' if self.is_blocked==1 else 'FREE'})")

        # Auto-advance image or route command
        if self.is_blocked == 1:
            if self.current_idx < len(self.image_files) - 1:
                self.current_idx += 1
                self.clicked_points = []
                self._check_initial_blocked_state()
                print("[*] Auto-advanced to NEXT IMAGE after BLOCKED mark.")
            else:
                print("[+] All images completed!")
        else:
            if curr_cmd == 'LEFT':
                self.active_cmd = 'STRAIGHT'
                self.clicked_points = []
                print("[*] Auto-switched Route Command to: STRAIGHT")
            elif curr_cmd == 'STRAIGHT':
                self.active_cmd = 'RIGHT'
                self.clicked_points = []
                print("[*] Auto-switched Route Command to: RIGHT")
            elif curr_cmd == 'RIGHT':
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                    self.active_cmd = 'LEFT'
                    self.clicked_points = []
                    self._check_initial_blocked_state()
                    print("[*] Completed image. Auto-advanced to NEXT IMAGE [LEFT]")
                else:
                    print("[+] All images completed!")

        self._draw()

    def run(self):
        print("\n" + "="*60)
        print(" JetRacer Multi-Task Trajectory & Free/Block Labeler GUI (Standard View)")
        print("="*60)
        print(" Controls:")
        print("  - Left Mouse Click : Click 5 Waypoints (W1 -> W2 -> W3 -> W4 -> W5)")
        print("  - Key 'b'          : Mark as BLOCKED (Auto-assigns safe WPs, saves & advances!)")
        print("  - Key 'f'          : Mark as FREE")
        print("  - Keys 1 / 2 / 3   : Switch Command (1: LEFT | 2: STRAIGHT | 3: RIGHT)")
        print("  - Key 's'          : Save JSON & Copy Image to Output Directory")
        print("  - Key 'r'          : Reset/Clear points for current image")
        print("  - Key 'n' / Right  : Next Image")
        print("  - Key 'p' / Left   : Previous Image")
        print("  - Key 'q' / Esc    : Quit GUI")
        print("="*60 + "\n")

        self._draw()

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in [ord('q'), 27]:
                print("[*] Exiting Labeler GUI.")
                break
            elif key == ord('1'):
                self.active_cmd = 'LEFT'
                print("[*] Route Command -> LEFT")
                self._draw()
            elif key == ord('2'):
                self.active_cmd = 'STRAIGHT'
                print("[*] Route Command -> STRAIGHT")
                self._draw()
            elif key == ord('3'):
                self.active_cmd = 'RIGHT'
                print("[*] Route Command -> RIGHT")
                self._draw()
            elif key == ord('b'):
                self.mark_blocked_and_save()
            elif key == ord('f'):
                self.is_blocked = 0
                print("[*] Safety Status -> FREE")
                self._draw()
            elif key == ord('s'):
                self.save_current()
            elif key == ord('r'):
                self.clicked_points = []
                print("[*] Reset waypoints for current image.")
                self._draw()
            elif key in [ord('n'), 83]:
                if self.current_idx < len(self.image_files) - 1:
                    self.current_idx += 1
                    self.clicked_points = []
                    self._check_initial_blocked_state()
                    self._draw()
            elif key in [ord('p'), 81]:
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.clicked_points = []
                    self._check_initial_blocked_state()
                    self._draw()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Task Trajectory & Free/Block Labeler GUI")
    parser.add_argument('--dataset_dir', type=str, default='urban_dataset', help='Primary raw image dataset folder')
    parser.add_argument('--move_controller_dir', type=str, default=r'd:\JetRacer_AI\jetracer\notebooks\move_controller\dataset', help='Move controller free/blocked dataset folder')
    parser.add_argument('--output_dir', type=str, default='multitask_dataset', help='Dedicated labeled output dataset folder')

    args = parser.parse_args()

    dirs = [args.dataset_dir, args.move_controller_dir]
    labeler = TrajectoryLabelerGUI(dataset_dirs=dirs, output_dir=args.output_dir)
    labeler.run()
