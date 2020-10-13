FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version
COPY requirements.txt requirements.txt
COPY nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc
RUN apt-get update && apt-get install -y libopencv-python && apt-get install -y --no-install-recommends \
          python3-pip \
          python3-dev \
          python3-tk \
          python3-matplotlib \
          python3-scipy \
          build-essential \
          zlib1g-dev \
          zip \
          libjpeg8-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install -U pip setuptools wheel
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose
RUN cd trt_pose && python3 setup.py install
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
RUN cd torch2trt && python3 setup.py install
RUN git clone https://github.com/IW276/IW276WS20-P12
