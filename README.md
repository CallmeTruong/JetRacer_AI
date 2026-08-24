# 🏎️ JetRacer AI — Autonomous Driving

Unified autonomous driving codebase for the **NVIDIA Jetson Nano** and **JetRacer** chassis. Migrated and refactored directly from `jetracer-car`. Integrates **Lane Following** (road tracking via ResNet-18 regression & Stanley controller) and an advanced **Smart City Urban Traffic System** (YOLO Object Detection, Obstacle Evasion State Machine, Conditioned Trajectory Prediction, and FSM Decision Logic).

---

## 🌟 Visual Model Predictions & Capabilities

### 🛣️ 1. Lane Following & Steering Vector Prediction
ResNet-18 regression model (`02_train_model_onnx.ipynb`) predicting target apex point `(x, y)` from camera frames, converted into steering angles by the **Stanley Controller**.

| Center Line Steering (`-0.06`) | Left Offset Steering (`-0.73`) | Right Offset Steering (`+0.89`) |
| :-: | :-: | :-: |
| ![Lane Pred Center](images/lane_pred_center.jpg) | ![Lane Pred Left](images/lane_pred_left.jpg) | ![Lane Pred Right](images/lane_pred_right.jpg) |

---

### 📍 2. Conditioned Trajectory Waypoints Model
Conditioned MobileNetV2 model (`best_trajectory_mobilenet.onnx`) predicting 5 route waypoints `(x, y)` based on high-level navigation commands (**Cmd: TURN LEFT**, **Cmd: STRAIGHT**, **Cmd: TURN RIGHT**).

| Cmd: TURN LEFT | Cmd: STRAIGHT | Cmd: TURN RIGHT |
| :-: | :-: | :-: |
| ![Trajectory Left](images/trajectory_cmd_turn_left.jpg) | ![Trajectory Straight](images/trajectory_cmd_straight.jpg) | ![Trajectory Right](images/trajectory_cmd_turn_right.jpg) |

---

### 🚦 3. Real-Time Object & Traffic Light Detection (YOLO)
YOLO detector (`best.onnx`) predicting traffic signals (**Green Light**, **Red Light**) and traffic signs (**Stop**, **Prohibition**, **Turn Left/Right**, **Straight Ahead**).

| Red Light Signal (Stop) | Green Light Signal (Go) |
| :-: | :-: |
| ![Red Light Detection](images/urban_red_light_annotated.jpg) | ![Green Light Detection](images/urban_green_light_annotated.jpg) |

---

### 🛑 4. Realtime Collision Avoidance
Binary MobileNet classifier monitoring path status to trigger emergency stop and reverse safety maneuvers.

| Path Blocked (Obstacle Detected) | Path Free (Clear Track) |
| :-: | :-: |
| ![Obstacle Blocked](images/obstacle_blocked.jpg) | ![Obstacle Free](images/obstacle_free.jpg) |

---

## 🔄 System Processing Workflows

### 🛣️ Workflow 1: Lane Following Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                 📷 CSI Camera Input                     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               🧠 ResNet-18 Regression                   │
└────────────────────────────┬────────────────────────────┘
                             │ (Target Point)
                             ▼
┌─────────────────────────────────────────────────────────┐
│                🎛️ Stanley Controller                    │
└────────────────────────────┬────────────────────────────┘
                             │ (Steering & Throttle)
                             ▼
┌─────────────────────────────────────────────────────────┐
│               🚗 Racecar HAL Execution                  │
└─────────────────────────────────────────────────────────┘
```

---

### 🅰️ Workflow 2: Smart City Option A Pipeline (FSM & Escape Maneuver)

```
                  ┌───────────────────────┐
                  │   📷 CSI Camera Input │
                  └───────────┬───────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │ 🔍 YOLO Sign Detector│      │🛑 MobileNet Classifier│
   └───────────┬──────────┘      └───────────┬──────────┘
               │                             │
               │ (Sign Detections)           ├───────────────┐ (Path Blocked)
               │                             ▼               │
               │                 ┌──────────────────────┐    │
               │                 │   🔄 Reverse Turning │    │
               │                 └───────────┬──────────┘    │
               │                             │               │
               │                             ▼               │
               │                 ┌──────────────────────┐    │
               │                 │       ⏸️ Pause        │    │
               │                 └───────────┬──────────┘    │
               │                             │               │
               │                             ▼               │
               │                 ┌──────────────────────┐    │
               │                 │   🔍 Check Forward   │    │
               │                 └───────────┬──────────┘    │
               │                             │ (Path Clear)  │
               ▼                             ▼               │
   ┌────────────────────────────────────────────────────┐    │
   │               🚦 Traffic FSM Engine                 │    │
   └───────────┬────────────────────────────────────────┘    │
               │                                             │
               ▼                                             ▼
   ┌────────────────────────────────────────────────────────────┐
   │              🚗 Racecar Controller Execution               │
   └────────────────────────────────────────────────────────────┘
```

---

### 🅱️ Workflow 3: Smart City Option B Pipeline (3-Model Multi-Task)

```
                              ┌───────────────────────┐
                              │   📷 CSI Camera Input │
                              └───────────┬───────────┘
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────────┐
│ 🔍 YOLO Sign/Light│            │🛑 MobileNet Safety│            │📍 Trajectory Model   │
└────────┬─────────┘            └────────┬─────────┘            └──────────┬───────────┘
         │                               │                                 │
         ▼                               │                                 ▼
┌──────────────────┐                     │ (Path Blocked)       ┌──────────────────────┐
│🚦 Traffic Rules  ├─────────────────────┼─────────────────────►│ 🎛️ Pure Pursuit Ctrl │
└──────────────────┘                     │                      └──────────┬───────────┘
                                         ▼                                 │
                            ┌──────────────────────────┐                   │ (Path Free)
                            │ 🔄 Escape State Machine  │                   │
                            │ (Reverse/Pause/Check)    │                   │
                            └────────────┬─────────────┘                   │
                                         │                                 │
                                         ▼                                 ▼
                            ┌──────────────────────────────────────────────────────────┐
                            │              🚗 Racecar Hardware Execution               │
                            └──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Complete Installation & Vehicle Setup Guide

Follow these step-by-step instructions to configure your Jetson Nano environment, ROS Melodic workspace, CUDA, TensorRT, and Python 3 dependencies.

### Step 1: Install System Environment & ROS Melodic
Run the automated environment setup script (or execute manually):

```bash
bash scripts/setup_env.sh
```

<details>
<summary><b>Click to view manual ROS & CUDA setup commands</b></summary>

```bash
# 1. Add ROS Melodic Repository & Keys
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# 2. Install ROS Core Packages & Dependencies
sudo apt update
sudo apt install -y ros-melodic-ros-base ros-melodic-catkin python-catkin-tools
sudo apt install -y ros-melodic-tf ros-melodic-tf2 ros-melodic-tf2-ros ros-melodic-gscam ros-melodic-web-video-server
sudo apt install -y python3-pip python3-dev python3-catkin-pkg python3-rospkg python3-empy python3-yaml python3-pycuda jupyterlab

pip3 install numpy catkin_pkg rospkg empy opencv-python requests

# 3. Configure CUDA & Python 3 rospy Environment in ~/.bashrc
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages' >> ~/.bashrc

source ~/.bashrc

# 4. Verify Python 3 rospy Import
python3 -c "import rospy; print('Python 3 rospy: OK')"
```
</details>

---

### Step 2: Build Vehicle Catkin Workspace
Initialize your Catkin workspace and build vehicle packages using Python 3:

```bash
bash scripts/setup_car.sh
```

<details>
<summary><b>Click to view manual Workspace Build commands</b></summary>

```bash
# 1. Initialize Workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace

# 2. Clone Vehicle Hardware & Driver Repositories
git clone https://github.com/waveshare/jetracer_ros.git
git clone https://github.com/NVIDIA-AI-IOT/jetracer.git

cd jetracer
sudo python3 setup.py install

# 3. Build Catkin Workspace with Python 3
cd ~/catkin_ws
rm -rf build devel
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash

# 4. Install JetRacer_AI Package in Editable Mode
cd ~/JetRacer_AI
pip3 install -e .
```
</details>

---

### Step 3: Launch Camera Stream & Web Video Server

#### 📷 Launch CSI Camera Node
```bash
bash scripts/launch_camera.sh
```
*(If camera hardware is stuck or unresponsive, restart daemon via: `sudo systemctl restart nvargus-daemon` or `bash scripts/launch_camera.sh --restart-daemon`)*

#### 🌐 View Camera Stream in Browser
Start `web_video_server` in another terminal:
```bash
bash scripts/launch_web_stream.sh
```
Open browser at: `http://<jetson-ip>:8080/stream_viewer?topic=/csi_cam_0/image_raw`

---

### Step 4: Export TensorRT Engines (`trtexec`)
Accelerate ONNX model inference using TensorRT FP16:

```bash
# Run helper script
bash scripts/export_tensorrt.sh models/urban_traffic/best.onnx models/urban_traffic/best.engine

# Or execute trtexec directly
/usr/src/tensorrt/bin/trtexec --onnx=models/urban_traffic/best.onnx --saveEngine=models/urban_traffic/best.engine --fp16
```

---

### Step 5: Launch JupyterLab
Start JupyterLab for interactive notebook execution:

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

---

## 🏎️ Autonomous Driving Execution Modes

The platform supports two core autonomous driving execution modes:

---

### 🛣️ Mode 1: Autonomous Lane Following
Autonomous road and track navigation powered by ResNet-18 regression and the **Stanley Controller**.

- **AI Model**: ResNet-18 Regression (`road_following_model.onnx` / `.engine`) predicting target apex coordinates `(x, y)`.
- **Steering Control**: **Stanley Controller** calculates steering angles and adaptive speed throttle based on target point offset.
- **Interactive Notebook**: `notebooks/1_lane_following/04_road_following_live.ipynb`
- **Headless Execution**:
  ```bash
  python apps/speed_track.py
  ```

---

### 🏙️ Mode 2: Smart City Urban Driving Options

The platform offers **two architecture options** for smart city autonomous navigation:

#### 🅰️ Option A: Smart City FSM & Escape Pipeline (`smart_city_fsm.ipynb`)
*(Direct migration from `jetracer-car/notebooks/smart_city.ipynb`)*

A deterministic state machine architecture with real-time obstacle evasion & traffic sign reaction.

- **AI Models (2 Models)**:
  1. **YOLO Sign Detector** (`models/urban_traffic/best.onnx`) — Detects traffic signs and signals.
  2. **MobileNet Road Safety Classifier** (`models/urban_traffic/best_model_mobilenet.onnx`) — Evaluates road status (`FREE` vs `BLOCKED`).
- **Obstacle Escape State Machine**:
  - **Normal Drive**: Evaluates road status probability. If blocked frames persist, transitions to reverse turning. Otherwise executes `TrafficFSM` actions (`FORWARD`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`, `U_TURN`).
  - **Reverse Turning**: Reverses vehicle with counter-steering.
  - **Pause**: Pauses brief duration to settle vehicle inertia.
  - **Check Forward**: Moves forward slowly to re-evaluate road condition. Returns to normal drive if clear.
- **Traffic FSM Spatial Filtering**: Filters detections by bounding box area and screen ROI.

```bash
# Standalone headless execution
python apps/smart_city.py
```

---

#### 🅱️ Option B: Multi-Model Multi-Task Pipeline (`smart_city_multitask.ipynb`)
An advanced 3-model parallel inference system combining object detection, safety classification, and waypoint tracking.

- **AI Models (3 Parallel Models)**:
  1. **YOLO Sign & Light Detector** (`best.onnx`) — Real-time detection of traffic lights (`red-light`, `green-light`) and signs.
  2. **MobileNet Collision Avoidance** (`best_model_mobilenet.onnx`) — Real-time road status classification (`FREE` vs `BLOCKED`).
  3. **Conditioned Trajectory Model** (`best_trajectory_mobilenet.onnx`) — Predicts 5 route waypoints `(x, y)` given navigation commands (`LEFT`, `RIGHT`, `STRAIGHT`).
- **Obstacle Escape State Machine**: Uses the exact same state machine as Option A (`DRIVE`, `REVERSE_TURNING`, `PAUSE`, `CHECK_FORWARD`) when the path is `BLOCKED`.
- **Trajectory Steering (Path FREE)**: When the road is `FREE`, steering is continuously calculated by the **Conditioned Trajectory Model (5 Waypoints)** + `PurePursuitController`.

---

## 📓 Complete Notebook Workflows

### Phase 1 — Lane Following (`notebooks/1_lane_following/`)
1. **`01_interactive_data_collection.ipynb`** — Drive manually & label target steering points.
2. **`02_train_model_onnx.ipynb`** — Train ResNet-18 regression model & export to ONNX.
3. **`03_export_tensorrt.ipynb`** — Convert ONNX model to TensorRT engine for Jetson GPU acceleration.
4. **`04_road_following_live.ipynb`** — Run real-time autonomous driving with Stanley control.

### Phase 2 — Urban Traffic (`notebooks/2_urban_traffic/`)
1. **`data_collection_raw.ipynb`** — Capture urban track images for training.
2. **`train_conditioned_lane_model.ipynb`** — Train direction-conditioned lane model.
3. **`collision_avoidance/`** — Collect data, train collision classifier & test emergency reverse.
4. **`export_tensorrt_engine.ipynb`** — Batch export YOLO and classification models to TensorRT.
5. **`smart_city_fsm.ipynb`** — Real-time urban driving with Option A (FSM + Escape pipeline).
6. **`smart_city_multitask.ipynb`** — Real-time urban driving with Option B (3-model pipeline).

---

## 📂 Project Structure

```
JetRacer_AI/
├── jetracer_ai/                  # Core Python package
│   ├── core/                     # ONNX engine, Pure Pursuit, base runner
│   ├── hardware/                 # Motor & steering HAL
│   ├── lane_following/           # Stanley controller & road runner
│   ├── urban_traffic/            # YOLO processor, FSM & intersection decisions
│   └── utils/                    # Camera stream & XY dataset helpers
├── notebooks/
│   ├── 1_lane_following/         # Data collection → Training → TRT Export → Live
│   └── 2_urban_traffic/          # Raw capture, FSM demo & multi-model pipeline
├── models/                       # ONNX, TensorRT, PyTorch weights
├── datasets/                     # Raw and labeled image captures
├── apps/                         # Standalone python scripts (smart_city, speed_track)
├── tools/                        # Annotation GUIs (bbox & trajectory)
├── scripts/                      # Setup & launch bash scripts
│   ├── setup_env.sh              # System, ROS, CUDA & PyCUDA setup
│   ├── setup_car.sh              # Workspace & vehicle driver build
│   ├── launch_camera.sh          # ROS CSI camera launcher
│   ├── launch_web_stream.sh      # Web video server stream launcher
│   └── export_tensorrt.sh        # trtexec TensorRT exporter
└── config/                       # Settings & gain parameters
```

---

## 🏷️ Annotation Tools

Desktop GUI utilities located in `tools/` for annotating collected datasets:

```bash
# Bounding box annotation for YOLO sign & traffic light detector
python tools/label_bbox_gui.py

# Trajectory waypoint labeling for direction model
python tools/label_trajectory_gui.py
```
**Note that you can using roboflow or any other tools to label your data (it's faster and have more features)**

---

## 🛠️ Configuration & Gains

Centralized parameters in [`config/settings.py`](config/settings.py):
- ROS camera topic (`/csi_cam_0/image_raw`) & MQTT broker settings
- Steering and throttle gain parameters (`BASE_THROTTLE`, `STEERING_GAIN`, PID parameters)
- Turn duration timing and safe zone percentages

---

## ❓ Troubleshooting

- **Camera Feed Black / Frozen**: Run `sudo systemctl restart nvargus-daemon` or `bash scripts/launch_camera.sh --restart-daemon`.
- **`rospy` Import Error**: Ensure `export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages` is in `~/.bashrc`.
- **Low Inference FPS**: Convert ONNX models to TensorRT engines using `bash scripts/export_tensorrt.sh`.
- **Import Failure**: Verify package installation by running `pip install -e .` in the project root.
