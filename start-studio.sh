#!/bin/bash
# Run the FramePack Studio interface

# Use default values if environment variables are not set
SERVER=${SERVER:-0.0.0.0}
PORT=${PORT:-7880}
SHARE=${SHARE:-false}

# Conditionally add share flag based on environment variable
SHARE_ARG=""
if [ "$SHARE" = "true" ]; then
    SHARE_ARG="--share"
fi

# Conditionally add inbrowser flag based on environment variable
INBROWSER_ARG=""
if [ "$INBROWSER" = "true" ]; then
    INBROWSER_ARG="--inbrowser"
fi

# Debug info
echo "Starting FramePack Studio with: server=$SERVER, port=$PORT, share=$SHARE, inbrowser=$INBROWSER"

# Run studio interface
python studio.py --server $SERVER --port $PORT $SHARE_ARG $INBROWSER_ARG