FROM ubuntu:latest

WORKDIR /app

# Define build argument
ARG MINICONDA_ARCH=aarch64

# Set environment variable for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    bash \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    curl \
    vim \
    build-essential \
    python3-dev \
    libhdf5-dev \
    pkg-config \
    ffmpeg \
    && apt-get clean

    # Conditional logic based on architecture
RUN ARCH=$(uname -m) && echo "Architecture detected: $ARCH"
      # x86_64   aarch64
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_22.11.1-1-Linux-${MINICONDA_ARCH}.sh \
            -O /tmp/anaconda.sh

RUN bash /tmp/anaconda.sh -b -p /opt/conda && \
      rm /tmp/anaconda.sh

      # Set environment variables for Conda
ENV PATH /opt/conda/bin:$PATH

# Initialize Conda
RUN conda init bash

# Run the shell to activate conda
SHELL ["/bin/bash", "-c"]

# Create conda environment with python 3.12
# RUN conda create -n myenv -c conda-forge python=3.7
# Activate environment in future commands (optional)
# RUN echo "conda activate myenv" >> ~/.bashrc

# RUN conda install -c conda-forge python=3.5
RUN pip install imutils
RUN pip install --user cython h5py
RUN pip install keras_tqdm
RUN pip install msgpack
# RUN pip install keras-vis
RUN conda clean --all
RUN conda update conda 
RUN conda install -n base conda-libmamba-solver
RUN pip install --user git+https://github.com/Theano/Theano.git
RUN pip install jupyter
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install pillow==8.3.2
RUN pip install opencv-python
RUN pip install tf-keras-vis
RUN conda install tensorflow=2.10.0 --solver=libmamba
RUN conda install keras=2.10.0
RUN pip install matplotlib

ENV DISPLAY=host.docker.internal:0
ENV QT_X11_NO_MITSHM=1