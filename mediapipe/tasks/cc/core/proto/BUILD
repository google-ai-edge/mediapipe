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

licenses(["notice"])

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_proto_library(
    name = "external_file_proto",
    srcs = ["external_file.proto"],
    visibility = [
        "//mediapipe/tasks:internal",
    ],
)

mediapipe_proto_library(
    name = "acceleration_proto",
    srcs = ["acceleration.proto"],
    visibility = [
        "//mediapipe/tasks:internal",
    ],
    deps = [
        "//mediapipe/calculators/tensor:inference_calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "base_options_proto",
    srcs = ["base_options.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":acceleration_proto",
        ":external_file_proto",
        "//mediapipe/gpu:gpu_origin_proto",
    ],
)

mediapipe_proto_library(
    name = "inference_subgraph_proto",
    srcs = ["inference_subgraph.proto"],
    deps = [
        ":base_options_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "model_resources_calculator_proto",
    srcs = ["model_resources_calculator.proto"],
    deps = [
        ":external_file_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)
