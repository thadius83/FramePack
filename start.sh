#!/bin/bash
# Startup script for FramePack using .env configuration

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

# Activate the virtual environment
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
  # Use . instead of source for better compatibility
  . .venv/bin/activate 2>/dev/null || {
    echo "Note: Using venv python directly instead of activation"
    PYTHON_CMD=".venv/bin/python"
  }
else
  echo "Warning: .venv directory or activate script not found. Please run ./install.sh first."
  exit 1
fi

# Set python command based on whether venv was activated
if [ -z "$PYTHON_CMD" ]; then
  # If VIRTUAL_ENV is set, activation worked
  if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="python"
  else
    # Fallback to direct path if activation didn't work properly
    PYTHON_CMD=".venv/bin/python"
  fi
fi

# Build CLI options
ARGS=""
if [ "$SHARE" = "true" ]; then
  ARGS="$ARGS --share"
fi
if [ "$INBROWSER" = "true" ]; then
  ARGS="$ARGS --inbrowser"
fi
ARGS="$ARGS --server $SERVER --port $PORT"

# Run the application
$PYTHON_CMD demo_gradio.py $ARGS