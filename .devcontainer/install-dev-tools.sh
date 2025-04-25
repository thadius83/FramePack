#!/bin/bash
# Script for installing development tools and dependencies

set -e

# update system
apt-get update
apt-get upgrade -y

# install Linux tools and add Python 3.10 repository
apt-get install -y software-properties-common wget curl git \
    build-essential libffi-dev gfortran \
    libjpeg-dev libpng-dev \
    libavfilter-dev libavformat-dev libavdevice-dev ffmpeg

# Add deadsnakes PPA and install Python 3.10
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils

# Create symbolic links to make python3.10 the default python3
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --set python3 /usr/bin/python3.10
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
update-alternatives --set python /usr/bin/python3.10

# Install pip for Python 3.10
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.10 get-pip.py
rm get-pip.py

# install Python packages
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirements.txt

# update CUDA Linux GPG repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb

# install cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
apt-get update
apt-get install -y libcudnn8=8.9.0.*-1+cuda11.8
apt-get install -y libcudnn8-dev=8.9.0.*-1+cuda11.8

# install recommended packages
apt-get install -y zlib1g g++ freeglut3-dev \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev

# clean up
pip3 cache purge
apt-get autoremove -y
apt-get clean