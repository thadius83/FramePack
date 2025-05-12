FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y     software-properties-common     wget     curl     git     build-essential     libffi-dev     gfortran     libjpeg-dev     libpng-dev     libavfilter-dev     libavformat-dev     libavdevice-dev     ffmpeg     zlib1g     g++     freeglut3-dev     libx11-dev     libxmu-dev     libxi-dev     libglu1-mesa     libglu1-mesa-dev     libfreeimage-dev     python3.10     python3.10-dev     python3.10-venv     python3.10-distutils     python-is-python3     && apt-get clean     && rm -rf /var/lib/apt/lists/*

# Set up Python and create symbolic links
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1     && update-alternatives --set python3 /usr/bin/python3.10     && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1     && update-alternatives --set python /usr/bin/python3.10

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py     && python3.10 get-pip.py     && rm get-pip.py

# Set up CUDA environment
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/usr/local/cuda/bin     LD_LIBRARY_PATH=:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64     XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip     && pip install --no-cache-dir --prefer-binary torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121     && pip install --no-cache-dir --prefer-binary -r requirements.txt     && pip install --no-cache-dir --prefer-binary torchvision

# Copy the entire application
COPY . /app/

# Set the default command to run both demos
CMD ["/bin/bash", "run-dual-demo.sh"]
