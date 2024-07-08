#!/bin/bash

# Ensure pre-commit is available
if ! command -v pre-commit &>/dev/null; then
    echo "pre-commit not found, attempting to install it..."
    pip install pre-commit
    if [ $? -ne 0 ]; then
        echo "Pre-commit install failed"
        exit 1
    fi
fi

# Install pre-commit hooks
pre-commit install

# Clone voice_type_classifier repository
git clone --recurse-submodules https://github.com/MarvinLvn/voice_type_classifier.git /workspaces/voice_type_classifier

# Download the latest Miniforge installer for macOS ARM64
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh

# Install coreutils if md5sum is not available
if ! command -v md5sum &> /dev/null; then
    echo "md5sum could not be found, installing coreutils..."
    sudo apt-get update
    sudo apt-get install -y coreutils
fi

# Create a symbolic link from md5 to md5sum if md5 is not available
if ! command -v md5 &> /dev/null && command -v md5sum &> /dev/null; then
    echo "Creating a symbolic link from md5 to md5sum..."
    sudo ln -s $(command -v md5sum) /usr/local/bin/md5
fi

# Run the installer
bash miniforge.sh -b -p $HOME/miniforge

# Initialize Conda
eval "$($HOME/miniforge/bin/conda shell.bash hook)"

# Activate the changes to .bashrc
source $HOME/.bashrc

# Remove the installer
rm miniforge.sh

# Add Miniforge to PATH
export PATH="$HOME/miniforge/bin:$PATH"

# Create Conda environment
cd /workspaces/voice_type_classifier && conda env create -f vtc.yml
if [ $? -ne 0 ]; then
    echo "Conda env creation failed"
    exit 1
fi

# Clone yolov5 repository
git clone https://github.com/ultralytics/yolov5.git /workspaces/yolov5

# Install YOLOv5 dependencies
pip install --upgrade pip \
    && pip install -r requirements.txt