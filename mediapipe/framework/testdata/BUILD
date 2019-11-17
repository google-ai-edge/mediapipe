#
# Copyright 2018 The MediaPipe Authors.
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

package(default_visibility = ["//visibility:private"])

exports_files(["LICENSE"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

proto_library(
    name = "sky_light_calculator_proto",
    srcs = ["sky_light_calculator.proto"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "sky_light_calculator_cc_proto",
    srcs = ["sky_light_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":sky_light_calculator_proto"],
)

proto_library(
    name = "night_light_calculator_proto",
    srcs = ["night_light_calculator.proto"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "night_light_calculator_cc_proto",
    srcs = ["night_light_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":night_light_calculator_proto"],
)

proto_library(
    name = "zoo_mutator_proto",
    srcs = ["zoo_mutator.proto"],
    deps = ["@com_google_protobuf//:any_proto"],
)

mediapipe_cc_proto_library(
    name = "zoo_mutator_cc_proto",
    srcs = ["zoo_mutator.proto"],
    cc_deps = ["@com_google_protobuf//:cc_wkt_protos"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":zoo_mutator_proto"],
)

proto_library(
    name = "zoo_mutation_calculator_proto",
    srcs = ["zoo_mutation_calculator.proto"],
    features = ["-proto_dynamic_mode_static_link"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [
        ":zoo_mutator_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:packet_factory_proto",
        "//mediapipe/framework:packet_generator_proto",
    ],
)
