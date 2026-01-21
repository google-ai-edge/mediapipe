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

cc_binary(
    name = "face_detection_full_range_cpu",
    data = ["//mediapipe/modules/face_detection:face_detection_full_range.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_detection:face_detection_full_range_desktop_live_deps",
    ],
)

# BUILD:
# bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
#   //mediapipe/examples/desktop/face_detection:face_detection_cpu
# RUN:
# bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu \
#   --calculator_graph_config_file=<.../face_detection_desktop_live.pbtxt> \
#   --input_video_path=<input video path> \
#   --output_video_path=<output video path>
cc_binary(
    name = "face_detection_cpu",
    data = ["//mediapipe/modules/face_detection:face_detection_short_range.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_detection:desktop_live_calculators",
    ],
)

# Linux only
cc_binary(
    name = "face_detection_gpu",
    data = ["//mediapipe/modules/face_detection:face_detection_short_range.tflite"],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/face_detection:desktop_live_gpu_calculators",
    ],
)
