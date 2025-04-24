#!/bin/bash
# Install script for FramePack

set -e

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
REQUIRED_PACKAGES="python3-venv python3-pip"

# Add Python version-specific packages
if [[ "$PYTHON_VERSION" == "3.10" ]]; then
  REQUIRED_PACKAGES="$REQUIRED_PACKAGES python3.10-venv"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
  REQUIRED_PACKAGES="$REQUIRED_PACKAGES python3.11-venv"
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
  REQUIRED_PACKAGES="$REQUIRED_PACKAGES python3.12-venv"
fi

# Add packages needed for building scipy and other dependencies
REQUIRED_PACKAGES="$REQUIRED_PACKAGES build-essential libffi-dev"

# Add packages needed for image processing (for cv2 and Pillow)
REQUIRED_PACKAGES="$REQUIRED_PACKAGES libjpeg-dev libpng-dev"

# Add packages needed for av (for video processing)
REQUIRED_PACKAGES="$REQUIRED_PACKAGES libavfilter-dev libavformat-dev libavdevice-dev ffmpeg"

# Try to install required packages
install_apt_packages "$REQUIRED_PACKAGES" || echo "Continuing without installing some required packages. Some features may not work."

# Try to install python3-venv specifically if needed
if ! $PYTHON_BIN -m venv --help &>/dev/null; then
  echo "Python venv module still not available. Attempting to install specific version..."
  install_apt_packages "python$PYTHON_VERSION-venv" || echo "Warning: Cannot install python-venv. Will proceed without virtual environment."
fi

# Check if virtual environment already exists
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
  echo "Using existing virtual environment..."
  source .venv/bin/activate
  PIP_CMD="pip"
  VENV_PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  echo "Virtual environment is using Python $VENV_PYTHON_VERSION"
  
  # Double-check compatibility
  if [[ ! "$VENV_PYTHON_VERSION" =~ ^3\.1[0-2]$ ]]; then
    echo "Warning: Virtual environment is using Python $VENV_PYTHON_VERSION, which may cause issues with SciPy."
    echo "Proceeding anyway, but installation might fail."
  fi
# Check if the python3-venv package is already installed
elif ! $PYTHON_BIN -m venv --help &>/dev/null; then
  echo "python3-venv is not available. Using system Python directly..."
  PIP_CMD="$PYTHON_BIN -m pip"
  VENV_PYTHON_VERSION=$PYTHON_VERSION
  echo "Using Python $VENV_PYTHON_VERSION directly"
else
  # We can create a virtual environment
  echo "Creating new Python virtual environment (.venv) with $PYTHON_BIN..."
  $PYTHON_BIN -m venv .venv || {
    echo "Failed to create virtual environment. Using system Python instead."
    PIP_CMD="$PYTHON_BIN -m pip"
    VENV_PYTHON_VERSION=$PYTHON_VERSION
    echo "Using Python $VENV_PYTHON_VERSION directly"
  }
  
  # If virtual environment was created successfully
  if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    # Activate the virtual environment
    echo "Activating virtual environment..."
    source .venv/bin/activate
    
    # Use the virtual environment's pip
    PIP_CMD="pip"
    
    # Verify the Python version in the virtual environment
    VENV_PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Virtual environment is using Python $VENV_PYTHON_VERSION"

    # Double-check compatibility
    if [[ ! "$VENV_PYTHON_VERSION" =~ ^3\.1[0-2]$ ]]; then
      echo "Warning: Virtual environment is using Python $VENV_PYTHON_VERSION, which may cause issues with SciPy."
      echo "Proceeding anyway, but installation might fail."
    fi
  fi
fi

# Upgrade pip
echo "Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install dependencies with a modified approach for scipy
echo "Installing dependencies with binary wheels..."
$PIP_CMD install --upgrade setuptools wheel

# Try to find a compatible scipy version
echo "Installing dependencies..."
if [[ "$VENV_PYTHON_VERSION" =~ ^3\.1[0-2]$ ]]; then
  # For Python 3.10-3.12, try scipy 1.11.3 first as it has better compatibility
  echo "Trying scipy 1.11.3 for Python $VENV_PYTHON_VERSION..."
  if $PIP_CMD install --prefer-binary scipy==1.11.3; then
    # Install other dependencies except scipy
    echo "Installing remaining dependencies..."
    $PIP_CMD install --prefer-binary -r <(grep -v '^scipy==' requirements.txt)
  else
    # If 1.11.3 fails, try the version from requirements.txt
    echo "Falling back to requirements.txt specified scipy version..."
    $PIP_CMD install --prefer-binary -r requirements.txt
  fi
else
  # For other Python versions, try the scipy version from requirements.txt directly
  echo "Using requirements.txt as specified..."
  $PIP_CMD install --prefer-binary -r requirements.txt
fi

# Copy .env.example to .env if .env does not exist
if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    echo "No .env file found. Copying .env.example to .env..."
    cp .env.example .env
    echo "Please review and edit .env as needed."
  else
    echo "No .env.example file found. Please create a .env file manually."
  fi
fi

echo "Installation complete. You can now run ./start.sh to launch the app."