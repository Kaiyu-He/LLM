# Use the official Ubuntu 20.04 image as your parent image.
FROM 210.75.240.150:30003/zxc/gpu:python_3.9.18_torch2.1.1_transformers_4.35.2_cuda_12.1_optimum

# Set the working directory within your container to /app.
WORKDIR /train

# Let the python output directly show in the terminal without buffering it first.
ENV PYTHONUNBUFFERED=1

# Update the list of packages, then install some necessary dependencies.
RUN apt-get update && apt-get install -y \
  wget \
  git \
  bzip2 \
  libglib2.0-0 \
  libxext6 \
  libsm6 \
  libxrender1 \
  make\
  g++ 

RUN rm -rf /var/lib/apt/lists/*

# Download and install the latest version of Miniconda to /opt/conda.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
  && rm Miniconda3-latest-Linux-x86_64.sh 

# Add Miniconda's binary directory to PATH.
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n train python=3.9.7
RUN conda init bash

# Activate the conda environment.
RUN echo "conda activate train" >> ~/.bashrc
ENV PATH /opt/conda/envs/train/bin:$PATH

RUN /bin/bash -c "source ~/.bashrc && pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html"
RUN /bin/bash -c "source ~/.bashrc && pip install wandb"
RUN /bin/bash -c "source ~/.bashrc && conda install -c nvidia cuda-compiler"
ENV CUDA_HOME=/opt/conda/envs/train
RUN git clone http://210.75.240.150:9980/hekaiyu/env.git
WORKDIR /train/env
RUN /bin/bash -c "source ~/.bashrc && pip install --no-cache-dir -r requirements.txt"
RUN /bin/bash -c "source ~/.bashrc && pip install flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
RUN /bin/bash -c "source ~/.bashrc && pip install deepspeed-0.14.2+cu121torch2.3-cp39-cp39-manylinux_2_24_x86_64.whl"
RUN /bin/bash -c "source ~/.bashrc && pip install ai2-olmo"
WORKDIR /train