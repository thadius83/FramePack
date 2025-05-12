#\!/bin/bash
# Post-creation script for FramePack DevContainer

set -e

echo "Setting up FramePack development environment..."

# Create directories if they don't exist
mkdir -p /app/models
mkdir -p /app/outputs
mkdir -p /app/hf_download

# Create convenience scripts
mkdir -p /usr/local/bin/scripts

echo '#\!/bin/bash
cd /app && python demo_gradio.py --server 0.0.0.0 --port 7880' > /usr/local/bin/scripts/start-app.sh

chmod +x /usr/local/bin/scripts/start-app.sh

# Add scripts to PATH and create aliases
echo 'export PATH="/usr/local/bin/scripts:${PATH}"' >> /etc/bash.bashrc
echo 'alias start-app="/usr/local/bin/scripts/start-app.sh"' >> /etc/bash.bashrc

# Create a welcome message
echo "echo '
Welcome to FramePack Development Environment\!

Available commands:
  start-app          - Start the Gradio app

Environment is ready to use.
'" >> /etc/bash.bashrc

echo "FramePack development environment is ready\!"
