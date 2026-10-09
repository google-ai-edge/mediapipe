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
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "base_options",
    hdrs = ["base_options.h"],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    alwayslink = 1,
)

cc_library(
    name = "base_options_converter",
    srcs = ["base_options_converter.cc"],
    hdrs = ["base_options_converter.h"],
    deps = [
        ":base_options",
        "//mediapipe/tasks/cc/core:base_options",
        "//mediapipe/tasks/cc/core:host_environment",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_test(
    name = "base_options_converter_test",
    srcs = ["base_options_converter_test.cc"],
    deps = [
        ":base_options",
        ":base_options_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/core:base_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mp_status",
    hdrs = ["mp_status.h"],
)

cc_library(
    name = "mp_logging_c_lib",
    srcs = ["mp_logging.cc"],
    hdrs = ["mp_logging.h"],
    deps = [
        ":common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:log_entry",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:log_sink_registry",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_test(
    name = "mp_logging_test",
    srcs = ["mp_logging_test.cc"],
    deps = [
        ":mp_logging_c_lib",
        "//mediapipe/framework/port:gtest",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mp_status_converter",
    srcs = ["mp_status_converter.cc"],
    hdrs = ["mp_status_converter.h"],
    deps = [
        ":mp_status",
        "@com_google_absl//absl/status",
    ],
)
