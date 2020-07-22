# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#==== ! Prerequisite ! ====
# $ sh mediapipe/examples/coral/setup.sh
#====

# for opencv 3.2 default
FROM ubuntu:18.04

MAINTAINER <mediapipe@google.com>

WORKDIR /mediapipe

ENV DEBIAN_FRONTEND=noninteractive

# Install MediaPipe & Coral deps

COPY update_sources.sh /
RUN /update_sources.sh

RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN apt-get update && apt-get install -y \
  build-essential \
  crossbuild-essential-arm64 \
  libusb-1.0-0-dev:arm64 \
  zlibc:arm64 \
  pkg-config \
  zip \
  unzip \
  curl \
  wget \
  git \
  python \
  python-pip \
  python3-pip \
  python-numpy \
  vim-common \
  ca-certificates \
  emacs \
  software-properties-common && \
  add-apt-repository -y ppa:openjdk-r/ppa && \
  apt-get update && apt-get install -y openjdk-8-jdk

RUN pip install --upgrade setuptools
RUN pip install future
RUN pip3 install six

COPY . /mediapipe/

# Install bazel
# Please match the current MediaPipe Bazel requirements according to docs.
ARG BAZEL_VERSION=2.0.0
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

# OpenCV (3.2 default in 18.04)

RUN apt-get update && apt-get install -y libopencv-dev

# Opencv libs copied from coral device into opencv32_arm64_libs

RUN cp opencv32_arm64_libs/* /usr/lib/aarch64-linux-gnu/.

# Edge tpu header and lib

RUN git clone https://github.com/google-coral/edgetpu.git /edgetpu
RUN cp /edgetpu/libedgetpu/direct/aarch64/libedgetpu.so.1.0 /usr/lib/aarch64-linux-gnu/libedgetpu.so

# See mediapipe/examples/coral/README.md to finish setup
