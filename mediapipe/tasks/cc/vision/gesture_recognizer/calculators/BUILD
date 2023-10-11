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

package(default_visibility = [
    "//mediapipe/app/xeno:__subpackages__",
    "//mediapipe/tasks:internal",
])

mediapipe_proto_library(
    name = "landmarks_to_matrix_calculator_proto",
    srcs = ["landmarks_to_matrix_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_proto",
    ],
)

cc_library(
    name = "handedness_to_matrix_calculator",
    srcs = ["handedness_to_matrix_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/tasks/cc/vision/gesture_recognizer:handedness_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "handedness_to_matrix_calculator_test",
    srcs = ["handedness_to_matrix_calculator_test.cc"],
    deps = [
        ":handedness_to_matrix_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "landmarks_to_matrix_calculator",
    srcs = ["landmarks_to_matrix_calculator.cc"],
    deps = [
        ":landmarks_to_matrix_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "landmarks_to_matrix_calculator_test",
    srcs = ["landmarks_to_matrix_calculator_test.cc"],
    deps = [
        ":landmarks_to_matrix_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/strings",
    ],
)

mediapipe_proto_library(
    name = "combined_prediction_calculator_proto",
    srcs = ["combined_prediction_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "combined_prediction_calculator",
    srcs = ["combined_prediction_calculator.cc"],
    deps = [
        ":combined_prediction_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

cc_test(
    name = "combined_prediction_calculator_test",
    srcs = ["combined_prediction_calculator_test.cc"],
    deps = [
        ":combined_prediction_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)
