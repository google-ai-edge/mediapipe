# Copyright 2020 Google LLC
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "instant_motion_tracking_deps",
    deps = [
        "//mediapipe/graphs/instant_motion_tracking/calculators:matrices_manager_calculator",
        "//mediapipe/graphs/instant_motion_tracking/calculators:sticker_manager_calculator",
        "//mediapipe/graphs/instant_motion_tracking/subgraphs:region_tracking",
        "//mediapipe/graphs/object_detection_3d/calculators:gl_animation_overlay_calculator",
    ],
)

mediapipe_binary_graph(
    name = "instant_motion_tracking_binary_graph",
    graph = "instant_motion_tracking.pbtxt",
    output_name = "instant_motion_tracking.binarypb",
    deps = [":instant_motion_tracking_deps"],
)
