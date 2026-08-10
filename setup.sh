#JetCar full setup
set -e

cd "$HOME"

echo "==== Installing JetCam===="
if [! -d "$HOME/jetcam"]; then
    git clone https://github.com/NVIDIA-AI-IOT/jetcam.git
fi

cd "$HOME/jetcam"
sudo python3 setup.py install

echo "==== Installing torch2trt===="
if [! -d "$HOME/torch2trt"]; then
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
fi

cd "$HOME/torch2trt"
sudo python3 setup.py install


echo "=== Installing JetRacer ==="
if [ ! -d "$HOME/jetracer" ]; then
    git clone https://github.com/NVIDIA-AI-IOT/jetracer.git
fi

cd "$HOME/jetracer"
sudo python3 setup.py install


echo "=== Setting Jetson Nano to 5W mode ==="
sudo nvpmodel -m1


echo ""
echo " JetRacer setup completed successfully"

