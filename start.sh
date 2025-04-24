#!/bin/bash
# Startup script for FramePack using .env configuration

# Load environment variables from .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Set defaults if not set in .env
: "${SERVER:=0.0.0.0}"
: "${PORT:=7880}"
: "${SHARE:=false}"
: "${INBROWSER:=false}"

# Activate the virtual environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "Warning: .venv directory not found. Please run ./install.sh first."
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
python demo_gradio.py $ARGS