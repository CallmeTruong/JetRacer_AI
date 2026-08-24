#!/bin/bash
set -e

cd "$HOME"
sudo apt-get update
sudo apt install opencv-python -y


echo "==== Installing JetCam ===="
if [ ! -d "$HOME/jetcam" ]; then
    git clone https://github.com/NVIDIA-AI-IOT/jetcam.git
fi

cd "$HOME/jetcam"
sudo python3 setup.py install


echo "=== Installing JetRacer ==="
if [ ! -d "$HOME/jetracer" ]; then
    git clone https://github.com/NVIDIA-AI-IOT/jetracer.git
fi

cd "$HOME/jetracer"
sudo python3 setup.py install

cd "$HOME"
echo "=== Installing Jupyter lab ==="
sudo apt-get install -y jupyterlab

echo "=== Setting Jetson Nano to 5W mode ==="
sudo nvpmodel -m1

echo ""
echo "JetRacer setup completed successfully"