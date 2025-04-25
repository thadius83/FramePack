#!/bin/bash
# Install script for FramePack in Docker environment

set -e

# Install CUDA and cuDNN dependencies
echo "Setting up NVIDIA CUDA and cuDNN dependencies..."
if command -v sudo &>/dev/null && [ "$(id -u)" -ne 0 ]; then
  # We're not root, use sudo
  sudo_cmd="sudo"
else
  # We're root or sudo is not available
  sudo_cmd=""
fi

# Check if CUDA keyring is already installed
if ! dpkg -l | grep -q cuda-keyring; then
  echo "Installing CUDA keyring..."
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
  $sudo_cmd dpkg -i cuda-keyring_1.0-1_all.deb
  rm cuda-keyring_1.0-1_all.deb
else
  echo "CUDA keyring already installed, skipping..."
fi

# Check if cuDNN repository is already set up
if [ ! -f /etc/apt/preferences.d/cuda-repository-pin-600 ]; then
  echo "Setting up cuDNN repository..."
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
  $sudo_cmd mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
  $sudo_cmd apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
  $sudo_cmd add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
  $sudo_cmd apt-get update
else
  echo "cuDNN repository already set up, skipping..."
  $sudo_cmd apt-get update
fi

# Check if cuDNN libraries are already installed
if ! dpkg -l | grep -q libcudnn8; then
  echo "Installing cuDNN libraries..."
  $sudo_cmd apt-get install -y libcudnn8=8.9.0.*-1+cuda11.8 libcudnn8-dev=8.9.0.*-1+cuda11.8 || {
    echo "Failed to install specific cuDNN version, trying generic install..."
    $sudo_cmd apt-get install -y libcudnn8 libcudnn8-dev
  }
else
  echo "cuDNN libraries already installed, skipping..."
fi

# Check if recommended packages are already installed
RECOMMENDED_PACKAGES="zlib1g g++ freeglut3-dev libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev"
MISSING_PACKAGES=""

for pkg in $RECOMMENDED_PACKAGES; do
  if ! dpkg -l | grep -q "^ii  $pkg "; then
    MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
  fi
done

if [ -n "$MISSING_PACKAGES" ]; then
  echo "Installing missing recommended packages: $MISSING_PACKAGES"
  $sudo_cmd apt-get install -y $MISSING_PACKAGES
else
  echo "All recommended packages already installed, skipping..."
fi

# Set CUDA environment variables
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Save CUDA environment variables to be sourced later
echo "# CUDA Environment Variables" > .cuda_env
echo "export PATH=\$PATH:/usr/local/cuda/bin" >> .cuda_env
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64" >> .cuda_env
echo "export XLA_FLAGS=\"--xla_gpu_cuda_data_dir=/usr/local/cuda\"" >> .cuda_env

# Function to check if Python version is between 3.10 and 3.12 (inclusive)
check_python_version() {
  local python_cmd=$1
  if ! command -v $python_cmd &> /dev/null; then
    return 1
  fi
  
  local version=$($python_cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  local major=$(echo $version | cut -d. -f1)
  local minor=$(echo $version | cut -d. -f2)
  
  if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
    return 0
  else
    return 1
  fi
}

# Find a compatible Python version (3.10-3.12)
for cmd in python3.10 python3.11 python3.12 python3 python; do
  if check_python_version $cmd; then
    PYTHON_BIN=$cmd
    PYTHON_VERSION=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Using $PYTHON_BIN (version $PYTHON_VERSION)"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Error: Could not find Python 3.10-3.12. Please install Python 3.10, 3.11, or 3.12."
  exit 1
fi

# Function to install apt packages with or without sudo
install_apt_packages() {
  local packages="$1"
  if command -v apt &>/dev/null; then
    if [ "$(id -u)" -eq 0 ]; then
      echo "Installing packages as root: $packages"
      apt update && apt install -y $packages
    elif command -v sudo &>/dev/null; then
      echo "Installing packages with sudo: $packages"
      sudo apt update && sudo apt install -y $packages
    else
      echo "Warning: Cannot install packages without sudo privileges."
      return 1
    fi
    return 0
  else
    echo "apt not found. Cannot install packages."
    return 1
  fi
}

# Install required system packages
echo "Checking for required system packages..."
REQUIRED_PACKAGES="python3-pip"

# Setup lists of required packages
BASE_PACKAGES="python3-pip python-is-python3"
DEV_PACKAGES=""
BUILD_PACKAGES="build-essential libffi-dev gfortran"
IMAGE_PACKAGES="libjpeg-dev libpng-dev"
VIDEO_PACKAGES="libavfilter-dev libavformat-dev libavdevice-dev ffmpeg"

# Add Python version-specific packages if needed
if [[ "$PYTHON_VERSION" == "3.10" ]]; then
  DEV_PACKAGES="$DEV_PACKAGES python3.10-dev"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
  DEV_PACKAGES="$DEV_PACKAGES python3.11-dev"
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
  DEV_PACKAGES="$DEV_PACKAGES python3.12-dev"
fi

# Function to check for missing packages
check_missing_packages() {
  local package_list="$1"
  local missing=""
  
  for pkg in $package_list; do
    if ! dpkg -l | grep -q "^ii  $pkg "; then
      missing="$missing $pkg"
    fi
  done
  
  echo "$missing"
}

# Check for missing packages in each category
MISSING_BASE=$(check_missing_packages "$BASE_PACKAGES")
MISSING_DEV=$(check_missing_packages "$DEV_PACKAGES")
MISSING_BUILD=$(check_missing_packages "$BUILD_PACKAGES")
MISSING_IMAGE=$(check_missing_packages "$IMAGE_PACKAGES")
MISSING_VIDEO=$(check_missing_packages "$VIDEO_PACKAGES")

# Combine all missing packages
MISSING_PACKAGES="$MISSING_BASE $MISSING_DEV $MISSING_BUILD $MISSING_IMAGE $MISSING_VIDEO"
MISSING_PACKAGES=$(echo "$MISSING_PACKAGES" | xargs)  # Trim whitespace

# Install only missing packages
if [ -n "$MISSING_PACKAGES" ]; then
  echo "Installing missing packages: $MISSING_PACKAGES"
  install_apt_packages "$MISSING_PACKAGES" || echo "Continuing without installing some required packages. Some features may not work."
else
  echo "All required packages already installed, skipping..."
fi

# Set up pip command
PIP_CMD="$PYTHON_BIN -m pip"
echo "Using Python $PYTHON_VERSION directly"

# Upgrade pip
echo "Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install dependencies with a modified approach for scipy
echo "Installing dependencies with binary wheels..."
$PIP_CMD install --upgrade setuptools wheel

# Try to find a compatible scipy version
echo "Installing dependencies..."
if [[ "$PYTHON_VERSION" =~ ^3\.1[0-2]$ ]]; then
  # For Python 3.10-3.12, try scipy 1.11.3 first as it has better compatibility, or fall back to 1.10.1
  echo "Trying scipy 1.11.3 for Python $PYTHON_VERSION..."
  if $PIP_CMD install --prefer-binary scipy==1.11.3; then
    # Install other dependencies except scipy
    echo "Installing remaining dependencies..."
    $PIP_CMD install --prefer-binary -r <(grep -v '^scipy==' requirements.txt)
  else
    echo "Trying scipy 1.10.1 as fallback..."
    if $PIP_CMD install --prefer-binary scipy==1.10.1; then
      # Install other dependencies except scipy
      echo "Installing remaining dependencies..."
      $PIP_CMD install --prefer-binary -r <(grep -v '^scipy==' requirements.txt)
    else
      # If both versions fail, try the version from requirements.txt
      echo "Falling back to requirements.txt specified scipy version..."
      $PIP_CMD install --prefer-binary -r requirements.txt
    fi
  fi
else
  # For other Python versions, try the scipy version from requirements.txt directly
  echo "Using requirements.txt as specified..."
  $PIP_CMD install --prefer-binary -r requirements.txt
fi

# Install torchvision (required by the app but missing from requirements.txt)
echo "Installing torchvision dependency..."
$PIP_CMD install --prefer-binary torchvision

# Check if python command works (it should with python-is-python3 package)
if ! command -v python &> /dev/null; then
  if command -v python3 &> /dev/null; then
    echo "Warning: 'python' command not found but 'python3' exists. python-is-python3 package might not be working."
    echo "Creating symlink for python -> python3 as a fallback..."
    $sudo_cmd ln -sf $(which python3) /usr/local/bin/python
  else
    echo "Warning: Neither 'python' nor 'python3' commands are available. Installation may fail."
  fi
else
  echo "Python command is properly configured."
fi

# Copy .env.example to .env if .env does not exist
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    echo "No .env file found. Copying .env.example to .env..."
    cp .env.example .env
    echo "Please review and edit .env as needed."
  else
    # Create a basic .env file with default settings
    echo "# Configuration for FramePack" > .env
    echo "SERVER=0.0.0.0" >> .env
    echo "PORT=7880" >> .env
    echo "SHARE=false" >> .env
    echo "INBROWSER=false" >> .env
    echo "MODEL_DIR=/path/to/your/models" >> .env
    echo "HF_TOKEN=" >> .env
    echo "No .env.example file found. Created a basic .env file. Please review and edit as needed."
  fi
fi

echo "Installation complete. You can now run:"
echo
echo "  ./start.sh                 # Launch the app"
echo