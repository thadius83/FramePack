#\!/bin/bash
# Script to start FramePack in DevPod environment

# Source environment variables
if [ -f .env ]; then
  # Use a safer approach to parse .env file
  while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    if [[ $line && \! $line =~ ^[[:space:]]*# ]]; then
      # Remove any trailing comments
      line=$(echo "$line"  < /dev/null |  sed 's/#.*$//')
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

# Build CLI options
ARGS=""
if [ "$SHARE" = "true" ]; then
  ARGS="$ARGS --share"
fi
if [ "$INBROWSER" = "true" ]; then
  ARGS="$ARGS --inbrowser"
fi
ARGS="$ARGS --server $SERVER --port $PORT"

# Start the application
echo "Starting FramePack on $SERVER:$PORT..."
python demo_gradio.py $ARGS
