cd
mkdir .mujoco
cd .mujoco
wget https://www.roboti.us/file/mjkey.txt
wget https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
rm -rf mjpro131_linux.zip
export MUJOCO_PY_MJKEY_PATH=~/.mujoco/mjkey.txt
export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro131
source ~/.bashrc
