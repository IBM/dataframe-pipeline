##############################################################################
# Copyright 2020 IBM Corp. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
##############################################################################

FROM ubuntu:18.04
MAINTAINER Takuya Nakaike "nakaike@jp.ibm.com"

ARG ONNXRUNTIME_REPO=https://github.com/tnakaike/onnxruntime
ARG ONNXRUNTIME_BRANCH=rel_1.3.1-pldi2021

RUN apt-get update && \
    apt-get install -y \
    sudo \
    git \
    bash \
    wget \
    zip \
    build-essential \
    emacs \
    python3.6 \
    python3.6-dev \
    python3-pip \
    libopenblas-dev \
    gfortran \
    protobuf-compiler \
    libprotobuf-dev \
    libssl-dev \
    python

WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1.tar.gz && \
    tar xfvz cmake-3.18.1.tar.gz && \
    cd cmake-3.18.1 && \
    ./bootstrap --parallel=8 && \
    make -j8 && \
    make install && \
    cd .. && \
    rm -rf cmake*

RUN mkdir -p /git
WORKDIR /git
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime

RUN pip3 install numpy && \
    cd onnxruntime && \
    ./build-python.sh && \
    cd .. && \
    rm -rf onnxruntime

RUN git clone https://github.com/IBM/dataframe-pipeline.git && \
    cd dataframe-pipeline && \
    pip3 install .

RUN mkdir -p /root/.kaggle
ADD kaggle.json /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json
