#!/bin/bash
# =====================================================================
# JetRacer AI — Web Video Stream Server Launcher
# Runs ROS web_video_server to view camera feed in browser at:
# http://<jetson-ip>:8080/stream_viewer?topic=/csi_cam_0/image_raw
# =====================================================================

if [ -f "/opt/ros/melodic/setup.bash" ]; then
    source "/opt/ros/melodic/setup.bash"
fi

if [ -f "$HOME/catkin_ws/devel/setup.bash" ]; then
    source "$HOME/catkin_ws/devel/setup.bash"
fi

echo -e "\e[1;34m[*] Starting ROS web_video_server...\e[0m"
echo -e "\e[1;32mOpen your browser at: http://<jetson-ip>:8080/stream_viewer?topic=/csi_cam_0/image_raw\e[0m\n"

rosrun web_video_server web_video_server
