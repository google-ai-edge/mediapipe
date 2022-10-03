# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

package(default_visibility = [
    "//mediapipe/app/xeno:__subpackages__",
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

mediapipe_proto_library(
    name = "hand_association_calculator_proto",
    srcs = ["hand_association_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "hand_association_calculator",
    srcs = ["hand_association_calculator.cc"],
    deps = [
        ":hand_association_calculator_cc_proto",
        "//mediapipe/calculators/util:association_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:rectangle_util",
    ],
    alwayslink = 1,
)

# TODO: Enable this test
