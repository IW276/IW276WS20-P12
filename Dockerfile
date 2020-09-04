FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
RUN echo "Build our Container based on L4T Pytorch"
RUN nvcc --version

#Example: 
RUN pip3 install -U \
        pip \
        setuptools \
        wheel && \
    pip3 install \
        -r requirements.txt \
         && \
    rm -rf ~/.cache/pip

# ....
