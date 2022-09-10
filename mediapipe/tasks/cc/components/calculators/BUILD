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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_proto_library(
    name = "classification_aggregation_calculator_proto",
    srcs = ["classification_aggregation_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "classification_aggregation_calculator",
    srcs = ["classification_aggregation_calculator.cc"],
    deps = [
        ":classification_aggregation_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/tasks/cc/components/containers:category_cc_proto",
        "//mediapipe/tasks/cc/components/containers:classifications_cc_proto",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "score_calibration_calculator_proto",
    srcs = ["score_calibration_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "score_calibration_calculator",
    srcs = ["score_calibration_calculator.cc"],
    deps = [
        ":score_calibration_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/tasks/cc:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

cc_test(
    name = "score_calibration_calculator_test",
    srcs = ["score_calibration_calculator_test.cc"],
    deps = [
        ":score_calibration_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "score_calibration_utils",
    srcs = ["score_calibration_utils.cc"],
    hdrs = ["score_calibration_utils.h"],
    deps = [
        ":score_calibration_calculator_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc:common",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "score_calibration_utils_test",
    srcs = ["score_calibration_utils_test.cc"],
    deps = [
        ":score_calibration_calculator_cc_proto",
        ":score_calibration_utils",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/tasks/metadata:metadata_schema_cc",
        "@com_google_absl//absl/strings",
    ],
)
