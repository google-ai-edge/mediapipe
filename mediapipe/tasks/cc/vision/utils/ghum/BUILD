# Copyright 2023 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "ghum_hand_topology",
    hdrs = ["ghum_hand_topology.h"],
)

cc_library(
    name = "ghum_hand_utils",
    srcs = ["ghum_hand_utils.cc"],
    hdrs = ["ghum_hand_utils.h"],
    deps = [
        ":ghum_hand_topology",
        "//mediapipe/calculators/util:set_joints_visibility_calculator",
        "//mediapipe/calculators/util:set_joints_visibility_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:body_rig_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_topology",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "ghum_topology",
    hdrs = ["ghum_topology.h"],
)

cc_library(
    name = "ghum_utils",
    srcs = ["ghum_utils.cc"],
    hdrs = ["ghum_utils.h"],
    deps = [
        ":ghum_topology",
        "//mediapipe/calculators/util:combine_joints_calculator",
        "//mediapipe/calculators/util:combine_joints_calculator_cc_proto",
        "//mediapipe/calculators/util:set_joints_visibility_calculator",
        "//mediapipe/calculators/util:set_joints_visibility_calculator_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/formats:body_rig_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_topology",
        "@com_google_absl//absl/types:span",
    ],
)
