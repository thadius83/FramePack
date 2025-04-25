#!/bin/bash
# Script to install optimization packages for FramePack

# Use set -x for debugging
set -x

echo "Installing optimization packages for FramePack..."

# Find the Python interpreter
if command -v python &> /dev/null; then
  PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
else
  echo "Error: Could not find Python. Please make sure Python is installed."
  exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using $PYTHON_CMD (version $PY_VERSION)"

# Try to use the existing PyTorch installation
echo "Checking existing PyTorch installation..."
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
  $PYTHON_CMD -c "import torch; print(f'Torch version: {torch.__version__}')"
  TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
  echo "Using existing PyTorch version: $TORCH_VERSION"
else
  echo "PyTorch is not properly installed. Try running the app without xformers optimization."
  exit 1
fi

CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'Not available')")
echo "CUDA version: $CUDA_VERSION"

# Install xformers
echo "Installing xformers for memory-efficient attention..."
$PYTHON_CMD -m pip install --upgrade pip

# Skip xformers installation due to potential compatibility issues
echo "Skipping xformers installation to avoid compatibility issues."
echo "The application will run with PyTorch's built-in optimizations instead."

# Install other optimizations if available
echo "Installing flash-attention if available for your system..."
$PYTHON_CMD -m pip install flash-attn --no-build-isolation || echo "Flash-attention not available for your system, skipping."

echo "Installing other optimization packages..."
$PYTHON_CMD -m pip install triton || echo "Triton installation failed, skipping."

# Install ninja for faster compilation
$PYTHON_CMD -m pip install ninja

echo "Checking if optimizations were successfully installed..."

# Check if xformers was installed
if $PYTHON_CMD -c "import xformers" 2>/dev/null; then
  echo "✅ xformers installed successfully!"
else
  echo "❌ xformers installation failed."
fi

# Check if flash-attention was installed
if $PYTHON_CMD -c "import flash_attn" 2>/dev/null; then
  echo "✅ flash-attention installed successfully!"
else
  echo "❌ flash-attention not installed."
fi

echo "Installation of optimization packages complete."
echo "You can now run './start-lowmem.sh' to use the optimized version."