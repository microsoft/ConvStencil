FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

LABEL maintainer="y_t_chen@outlook.com"

RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/cmake /usr/local/bin/cmake

COPY . /convStencil
WORKDIR /convStencil