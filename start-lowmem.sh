#!/bin/bash
# Startup script for FramePack in Docker environment with low memory optimizations

# Load environment variables from .env if it exists
if [ -f .env ]; then
  # Use a safer approach to parse .env file
  while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    if [[ $line && ! $line =~ ^[[:space:]]*# ]]; then
      # Remove any trailing comments
      line=$(echo "$line" | sed 's/#.*$//')
      # Export the variable
      export "$line"
    fi
  done < .env
fi

# Set defaults if not set in .env
: "${SERVER:=0.0.0.0}"
: "${PORT:=7880}"
: "${SHARE:=false}"
: "${INBROWSER:=false}"

# Load CUDA environment variables if available
if [ -f ".cuda_env" ]; then
  echo "Loading CUDA environment variables..."
  . .cuda_env
fi

# Find the Python interpreter
if command -v python &> /dev/null; then
  version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  major=$(echo $version | cut -d. -f1)
  minor=$(echo $version | cut -d. -f2)
  
  if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
    PYTHON_CMD="python"
    echo "Using $PYTHON_CMD (version $version)"
  else
    echo "Warning: Found Python version $version, which is outside the recommended range (3.10-3.12)."
    echo "Some features may not work correctly. Proceeding anyway..."
    PYTHON_CMD="python"
  fi
else
  # Fall back to python3
  for cmd in python3.10 python3.11 python3.12 python3; do
    if command -v $cmd &> /dev/null; then
      version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
      PYTHON_CMD=$cmd
      echo "Using $PYTHON_CMD (version $version)"
      break
    fi
  done
fi

if [ -z "$PYTHON_CMD" ]; then
  echo "Error: Could not find Python 3. Please make sure Python is installed."
  exit 1
fi

# Check if required packages are installed
echo "Checking if required Python packages are installed..."
if ! $PYTHON_CMD -c "import gradio" &>/dev/null; then
  echo "Warning: gradio package not found. Please run install-docker.sh first."
  exit 1
fi

# Clear CUDA cache before starting
echo "Clearing CUDA cache for better memory management..."
$PYTHON_CMD -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || echo "Could not clear CUDA cache (torch not available)"

# Build CLI options
ARGS="--low_memory"
if [ "$SHARE" = "true" ]; then
  ARGS="$ARGS --share"
fi
if [ "$INBROWSER" = "true" ]; then
  ARGS="$ARGS --inbrowser"
fi
ARGS="$ARGS --server $SERVER --port $PORT"

# Print memory status before running
echo "Available GPU memory before starting:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# Run the application with low memory optimizations
echo "Starting FramePack with low memory optimizations..."
$PYTHON_CMD demo_gradio_lowmem.py $ARGS