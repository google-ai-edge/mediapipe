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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/template_matching:template_matching_tflite
# RUN:
# bazel-bin/mediapipe/examples/desktop/template_matching/template_matching_tflite \
#   --calculator_graph_config_file=<.../template_matching_desktop.pbtxt> \
#   --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
cc_binary(
    name = "template_matching_tflite",
    deps = [
        "//mediapipe/examples/desktop:simple_run_graph_main",
        "//mediapipe/graphs/template_matching:desktop_calculators",
    ],
)
