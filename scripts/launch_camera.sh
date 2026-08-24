#!/bin/bash
# =====================================================================
# JetRacer AI — Camera Launch Script
# Starts roscore, restarts nvargus-daemon if needed, and launches
# CSI camera ROS node (/csi_cam_0/image_raw).
# =====================================================================

# 1. Source ROS & Workspace
if [ -f "/opt/ros/melodic/setup.bash" ]; then
    source "/opt/ros/melodic/setup.bash"
fi

export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages

CATKIN_SETUP="$HOME/catkin_ws/devel/setup.bash"
if [ -f "$CATKIN_SETUP" ]; then
    source "$CATKIN_SETUP"
    echo -e "\e[1;32m[OK] Sourced ROS Workspace: $CATKIN_SETUP\e[0m"
else
    echo -e "\e[1;31m[ERROR] Catkin workspace setup not found at $CATKIN_SETUP\e[0m"
    echo "Please run bash scripts/setup_car.sh first."
    exit 1
fi

# 2. Restart nvargus-daemon if camera hardware service is stuck
if [ "$1" == "--restart-daemon" ] || [ "$1" == "-r" ]; then
    echo -e "\e[1;33m[*] Restarting nvargus-daemon camera service...\e[0m"
    sudo systemctl restart nvargus-daemon
    sleep 2
fi

# 3. Start roscore if not running
if ! pgrep -x "roscore" > /dev/null && ! pgrep -x "rosmaster" > /dev/null; then
    echo -e "\e[1;33m[*] Starting ROS Master (roscore)...\e[0m"
    roscore &
    sleep 3
fi

# 4. Launch CSI Camera ROS Node
echo -e "\n\e[1;34m--- STARTING CSI CAMERA NODE ---\e[0m"
roslaunch jetracer csi_camera.launch &
CAMERA_PID=$!

echo -e "\n\e[1;32m=== CAMERA IS RUNNING ===\e[0m"
echo "Camera PID: $CAMERA_PID"
echo "Topic: /csi_cam_0/image_raw"
echo -e "\e[1;33mPress Ctrl+C to stop Camera.\e[0m"

# Handle graceful shutdown
trap "echo -e '\nStopping Camera...'; kill $CAMERA_PID 2>/dev/null; killall -9 nvargus_daemon_client jetson_camera 2>/dev/null; exit" SIGINT SIGTERM

wait
