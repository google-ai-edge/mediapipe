# Copyright 2021 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/selfie_segmentation:selfie_segmentation_cpu
# RUN:
# bazel-bin/mediapipe/examples/desktop/selfie_segmentation/selfie_segmentation_cpu \
#   --calculator_graph_config_file=<.../selfie_segmentation_cpu.pbtxt> \
#   --input_video_path=<input video path> \
#   --output_video_path=<output video path>
cc_binary(
    name = "selfie_segmentation_cpu",
    data = ["//mediapipe/modules/selfie_segmentation:selfie_segmentation.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/selfie_segmentation:selfie_segmentation_cpu_deps",
    ],
)

# Linux only
cc_binary(
    name = "selfie_segmentation_gpu",
    data = ["//mediapipe/modules/selfie_segmentation:selfie_segmentation.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/selfie_segmentation:selfie_segmentation_gpu_deps",
    ],
)
