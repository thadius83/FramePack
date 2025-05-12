#!/bin/bash
# Start the FramePack Studio with docker-compose

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Source the .env file if it exists
if [ -f .env ]; then
    source .env
fi

# Set HF_CACHE_DIR if not already set
if [ -z "$HF_CACHE_DIR" ]; then
    export HF_CACHE_DIR="./hf_download"
fi

# Start the service
docker-compose up -d framepack-studio

echo "FramePack Studio is starting on http://localhost:7882"
echo "View logs with: docker-compose logs -f framepack-studio"