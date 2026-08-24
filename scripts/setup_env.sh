#!/bin/bash
# =====================================================================
# JetRacer AI — System Environment & Dependencies Installer
# Installs: ROS Melodic, CUDA & PyCUDA, TensorRT, GStreamer, JupyterLab,
# and configures Python 3 ROS environment (rospy fix).
# =====================================================================

set -e

echo -e "\e[1;34m[1/6] Adding ROS Melodic Repository & Keys...\e[0m"
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 || true

sudo apt update

echo -e "\e[1;34m[2/6] Installing ROS Melodic & Core Packages...\e[0m"
sudo apt install -y \
    ros-melodic-ros-base \
    ros-melodic-catkin \
    python-catkin-tools \
    ros-melodic-tf \
    ros-melodic-tf2 \
    ros-melodic-tf2-ros \
    ros-melodic-gscam \
    ros-melodic-web-video-server \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

echo -e "\e[1;34m[3/6] Installing Python 3 ROS Dependencies & JupyterLab...\e[0m"
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-catkin-pkg \
    python3-rospkg \
    python3-empy \
    python3-yaml \
    python3-pycuda \
    jupyterlab \
    opencv-python

pip3 install --upgrade pip
pip3 install numpy catkin_pkg rospkg empy opencv-python requests

echo -e "\e[1;34m[4/6] Installing TensorRT & Developer Tools...\e[0m"
sudo apt-get install -y tensorrt python3-libnvinfer-dev libnvinfer-samples || true

echo -e "\e[1;34m[5/6] Setting Environment Variables in ~/.bashrc...\e[0m"

# ROS Melodic Sourcing
if ! grep -q "ros/melodic/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
fi

# CUDA Path
if ! grep -q "cuda/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi

# Python3 rospy Import Fix
if ! grep -q "PYTHONPATH.*/opt/ros/melodic/lib/python2.7/dist-packages" ~/.bashrc; then
    echo 'export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages' >> ~/.bashrc
fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages

echo -e "\e[1;34m[6/6] Verifying Python 3 rospy Import...\e[0m"
python3 -c "import rospy; print('[OK] Python 3 rospy import successful!')"

echo -e "\n\e[1;32m=== Environment Setup Completed Successfully! ===\e[0m"
