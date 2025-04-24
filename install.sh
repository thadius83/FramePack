#!/bin/bash
# Install script for FramePack

set -e

# Check for Python 3.10
PYTHON_BIN="python3"
PYTHON_VERSION=$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$PYTHON_VERSION" != "3.10" ]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
    PYTHON_VERSION=$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
  else
    echo "Warning: Python 3.10 is recommended for this project. Detected version: $PYTHON_VERSION"
    echo "Please install Python 3.10 and rerun this script for best compatibility."
  fi
fi

# Create Python virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating Python 3.10 virtual environment (.venv)..."
  $PYTHON_BIN -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Copy .env.example to .env if .env does not exist
if [ ! -f ".env" ]; then
  echo "No .env file found. Copying .env.example to .env..."
  cp .env.example .env
  echo "Please review and edit .env as needed."
fi

echo "Installation complete. You can now run ./start.sh to launch the app."