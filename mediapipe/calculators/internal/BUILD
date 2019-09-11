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

licenses(["notice"])  # Apache 2.0

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

package(default_visibility = ["//visibility:private"])

proto_library(
    name = "callback_packet_calculator_proto",
    srcs = ["callback_packet_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "callback_packet_calculator_cc_proto",
    srcs = ["callback_packet_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":callback_packet_calculator_proto"],
)

cc_library(
    name = "callback_packet_calculator",
    srcs = ["callback_packet_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":callback_packet_calculator_cc_proto",
        "//mediapipe/framework:calculator_base",
        "//mediapipe/framework:calculator_registry",
        "//mediapipe/framework:output_side_packet",
    ],
    alwayslink = 1,
)
