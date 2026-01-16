# Copyright 2022 The MediaPipe Authors.
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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

mediapipe_proto_library(
    name = "hand_landmarks_detector_graph_options_proto",
    srcs = ["hand_landmarks_detector_graph_options.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
    ],
)

mediapipe_proto_library(
    name = "hand_landmarker_graph_options_proto",
    srcs = ["hand_landmarker_graph_options.proto"],
    deps = [
        ":hand_landmarks_detector_graph_options_proto",
        ":hand_roi_refinement_graph_options_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_proto",
    ],
)

mediapipe_proto_library(
    name = "hand_roi_refinement_graph_options_proto",
    srcs = ["hand_roi_refinement_graph_options.proto"],
    deps = [
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
    ],
)

# TODO: open source hand joints graph
