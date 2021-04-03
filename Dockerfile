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

FROM ubuntu:18.04

MAINTAINER <mediapipe@google.com>

WORKDIR /io
WORKDIR /mediapipe

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        wget \
        unzip \
        python3-dev \
        python3-pip \
        python3-opencv \
        libopencv-core-dev \
        libopencv-highgui-dev \
        libopencv-imgproc-dev \
        libopencv-video-dev \
        libopencv-calib3d-dev \
        libopencv-features2d-dev \
        software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && apt-get install -y openjdk-8-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -O /mediapipe/BBB.mp4 https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4

RUN pip3 install --upgrade setuptools
RUN pip3 install future
RUN pip3 install six==1.14.0
RUN pip3 install wheel
RUN pip3 install tf_slim 
RUN pip3 install tensorflow==1.14.0

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install bazel
ARG BAZEL_VERSION=3.7.2
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget --no-check-certificate -O  /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -f /bazel/installer.sh

COPY . /mediapipe/

####
#### BUILD
####

#object detection TF
RUN bazel build -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --define no_aws_support=true \
    --linkopt=-s \
    mediapipe/examples/desktop/object_detection:object_detection_tensorflow
#object detection TFLite
RUN bazel build -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/object_detection:object_detection_tflite
#media sequence
RUN bazel build -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/media_sequence:media_sequence_demo
#autoflip
RUN bazel build -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/autoflip:run_autoflip

#yt8m
RUN mkdir -p /tmp/mediapipe
WORKDIR /tmp/mediapipe
RUN curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
RUN curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
RUN curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
RUN curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb

RUN curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
RUN tar -xzf inception-2015-12-05.tgz
RUN curl -O http://data.yt8m.org/models/baseline/saved_model.tar.gz
RUN tar -xf saved_model.tar.gz

WORKDIR /mediapipe
RUN bazel build -c opt \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --linkopt=-s \
    --define no_aws_support=true \
    mediapipe/examples/desktop/youtube8m:extract_yt8m_features
RUN bazel build -c opt \
    --define='MEDIAPIPE_DISABLE_GPU=1' \
    --linkopt=-s \
    mediapipe/examples/desktop/youtube8m:model_inference
RUN bazel build -c opt \
    --define='MEDIAPIPE_DISABLE_GPU=1' \
    --linkopt=-s \
    mediapipe/examples/desktop/youtube8m:model_inference


####
#### RUN
####

#object detection TF
RUN GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tensorflow \
    --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tensorflow_graph.pbtxt \
    --input_side_packets=input_video_path=mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/tmp/output-tensorflow.mp4
#object detection TFLite
RUN GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite \
    --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tflite_graph.pbtxt \
    --input_side_packets=input_video_path=mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/tmp/output-tflite.mp4

#media sequence
#TODO add python
RUN python -m mediapipe.examples.desktop.media_sequence.demo_dataset \
    --path_to_demo_data=/tmp/demo_data/ \
    --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/media_sequence/media_sequence_demo \
    --path_to_graph_directory=mediapipe/graphs/media_sequence/
RUN PYTHONPATH=$PYTHONPATH:/mediapipe python ./mediapipe/examples/desktop/media_sequence/read_demo_dataset.py

#autoflip
RUN GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip \
    --calculator_graph_config_file=mediapipe/examples/desktop/autoflip/autoflip_graph.pbtxt \
    --input_side_packets=input_video_path=mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/tmp/output-autoflip.mp4,aspect_ratio=1:1

#yt8m
RUN python3 -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
RUN python3 -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
    --path_to_input_video=/mediapipe/BBB.mp4 \
    --clip_end_time_sec=120

RUN GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
    --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
    --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  \
    --output_side_packets=output_sequence_example=/tmp/mediapipe/features.pb

# If we want the docker image to contain the pre-built object_detection_offline_demo binary, do the following
# RUN bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/demo:object_detection_tensorflow_demo

