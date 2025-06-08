ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ENV LANG C.UTF-8
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt -y upgrade && \
    apt-get -y install software-properties-common apt-utils git \
    build-essential yasm nasm cmake unzip wget curl \
    libtcmalloc-minimal4 pkgconf autoconf libtool libc6 libc6-dev \
    libnuma1 libnuma-dev libx264-dev libx265-dev libmp3lame-dev \
    python3-pip python3.12-dev python3-numpy && \
    ln -s /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Build nvidia codec headers
RUN git clone --depth=1 --branch=n13.0.19.0 \
    --single-branch https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && make install && \
    cd .. && rm -rf nv-codec-headers

# Build FFmpeg with NVENC support
RUN git clone --depth=1 --branch=n7.1.1 --single-branch https://github.com/FFmpeg/FFmpeg.git && \
    cd FFmpeg && \
    mkdir ffmpeg_build && cd ffmpeg_build && \
    ../configure \
    --enable-nonfree \
    --enable-cuda \
    --enable-libnpp \
    --enable-cuvid \
    --enable-ffnvcodec \
    --enable-nvdec \
    --enable-nvenc \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-gpl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libmp3lame \
    --extra-libs=-lpthread \
    --nvccflags="-arch=sm_75 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_89,code=sm_89 \
    -gencode=arch=compute_90,code=sm_90" && \
    make -j$(nproc) && make install && ldconfig && \
    cd ../.. && rm -rf FFmpeg

RUN mkdir /tmp/opencv && cd /tmp/opencv && \
    wget https://github.com/opencv/opencv/archive/4.11.0.zip -O opencv-4.11.0.zip && \
    unzip opencv-4.11.0.zip && cd opencv-4.11.0 && \
    wget https://github.com/opencv/opencv_contrib/archive/4.11.0.zip -O opencv_contrib-4.11.0.zip && \
    unzip opencv_contrib-4.11.0.zip && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release\
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_ZLIB=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D ENABLE_PRECOMPILED_HEADERS=OFF \
    -D WITH_OPENCL=OFF \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=OFF \
    -D WITH_CUDA=ON \
    -D WITH_GTK=OFF \
    -D WITH_OPENEXR=OFF \
    -D WITH_PROTOBUF=OFF \
    -D BUILD_LIST=python3,core,imgproc,imgcodecs,videoio,video,calib3d,flann,cudev,cudacodec \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv/opencv-4.11.0/opencv_contrib-4.11.0/modules/ .. && \
    make -j$(nproc) && make install && ldconfig && cd && rm -r /tmp/opencv

# Install PyTorch
RUN pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22 torchaudio==2.7.0 torchcodec==0.4.0 --index-url\
    https://download.pytorch.org/whl/cu128

# Main system requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
