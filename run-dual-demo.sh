#\!/bin/bash
# Run both demo scripts with different ports

# Use default values if environment variables are not set
SERVER=${SERVER:-0.0.0.0}
PORT1=${PORT:-7880}
PORT2=${PORT_F1:-7881}

# Run both demos
python demo_gradio.py --server $SERVER --port $PORT1 &
python demo_gradio_f1.py --server $SERVER --port $PORT2 &

# Keep container running
wait
