#\!/bin/bash
# Post-creation script for FramePack DevContainer

set -e

echo "Setting up FramePack development environment..."

# Create directories if they don't exist
mkdir -p /app/models
mkdir -p /app/outputs
mkdir -p /app/hf_download

# Create .env file from example if it doesn't exist
if [ ! -f /app/.env ]; then
  echo "Creating .env file from .env.example..."
  cp /app/.env.example /app/.env
  chmod 644 /app/.env
fi

# Create convenience scripts
mkdir -p /usr/local/bin/scripts

echo '#\!/bin/bash
cd /app && python demo_gradio.py --server 0.0.0.0 --port 7880' > /usr/local/bin/scripts/start-framepack.sh

echo '#\!/bin/bash
cd /app && python demo_gradio_f1.py --server 0.0.0.0 --port 7881' > /usr/local/bin/scripts/start-framepack-f1.sh

echo '#\!/bin/bash
cd /app && ./run-dual-demo.sh' > /usr/local/bin/scripts/start-dual-demo.sh

chmod +x /usr/local/bin/scripts/start-framepack.sh
chmod +x /usr/local/bin/scripts/start-framepack-f1.sh
chmod +x /usr/local/bin/scripts/start-dual-demo.sh

# Add scripts to PATH and create aliases
echo 'export PATH="/usr/local/bin/scripts:${PATH}"' >> /etc/bash.bashrc
echo 'alias start-framepack="/usr/local/bin/scripts/start-framepack.sh"' >> /etc/bash.bashrc
echo 'alias start-framepack-f1="/usr/local/bin/scripts/start-framepack-f1.sh"' >> /etc/bash.bashrc
echo 'alias start-dual-demo="/usr/local/bin/scripts/start-dual-demo.sh"' >> /etc/bash.bashrc

# Create a welcome message
echo "echo '
Welcome to FramePack Development Environment\!

Available commands:
  start-framepack    - Start FramePack Gradio app on port 7880
  start-framepack-f1 - Start FramePack-F1 Gradio app on port 7881
  start-dual-demo    - Run both FramePack and FramePack-F1 simultaneously

Environment is ready to use.
'" >> /etc/bash.bashrc

echo "FramePack development environment is ready\!"
