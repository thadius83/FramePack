#!/bin/bash
# Run the FramePack Studio interface

# Use default values if environment variables are not set
SERVER=${SERVER:-0.0.0.0}
PORT=${PORT:-7880}
SHARE=${SHARE:-false}

# Run studio interface
python studio.py --server $SERVER --port $PORT --share $SHARE