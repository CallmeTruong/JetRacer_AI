#!/bin/bash
# =====================================================================
# JetRacer AI — Catkin Workspace & Car Hardware Setup
# Creates catkin_ws, clones jetracer_ros & jetracer driver, builds
# workspace with Python 3, and installs jetracer_ai package.
# =====================================================================

set -e

SOURCE_ROS="/opt/ros/melodic/setup.bash"
if [ -f "$SOURCE_ROS" ]; then
    source "$SOURCE_ROS"
fi

CATKIN_DIR="$HOME/catkin_ws"
SRC_DIR="$CATKIN_DIR/src"

echo -e "\e[1;34m[1/4] Initializing Catkin Workspace at $CATKIN_DIR...\e[0m"
mkdir -p "$SRC_DIR"
cd "$SRC_DIR"

if [ ! -f "$SRC_DIR/CMakeLists.txt" ]; then
    catkin_init_workspace
fi

echo -e "\e[1;34m[2/4] Cloning Vehicle ROS Packages & Libraries...\e[0m"
if [ ! -d "$SRC_DIR/jetracer_ros" ]; then
    git clone https://github.com/waveshare/jetracer_ros.git "$SRC_DIR/jetracer_ros"
fi

if [ ! -d "$SRC_DIR/jetracer" ]; then
    git clone https://github.com/NVIDIA-AI-IOT/jetracer.git "$SRC_DIR/jetracer"
fi

cd "$SRC_DIR/jetracer"
sudo python3 setup.py install || python3 setup.py install --user

echo -e "\e[1;34m[3/4] Building Catkin Workspace with Python 3...\e[0m"
cd "$CATKIN_DIR"
rm -rf build devel
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

if ! grep -q "catkin_ws/devel/setup.bash" ~/.bashrc; then
    echo "source $CATKIN_DIR/devel/setup.bash" >> ~/.bashrc
fi

echo -e "\e[1;34m[4/4] Installing JetRacer_AI Package in Editable Mode...\e[0m"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
pip3 install -e .

echo -e "\n\e[1;32m=== Catkin Workspace & Vehicle Setup Completed! ===\e[0m"
