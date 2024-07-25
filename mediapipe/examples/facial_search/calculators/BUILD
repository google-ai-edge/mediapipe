# Copyright 2020 The MediaPipe Authors.
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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "closest_embeddings_calculator_proto",
    srcs = ["closest_embeddings_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "closest_embeddings_calculator_cc_proto",
    srcs = ["closest_embeddings_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":closest_embeddings_calculator_proto"],
)

cc_library(
    name = "closest_embeddings_calculator",
    srcs = ["closest_embeddings_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":closest_embeddings_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)
