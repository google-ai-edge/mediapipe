# Copyright 2025 The MediaPipe Authors.
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
    name = "common",
    hdrs = ["common.h"],
)

cc_library(
    name = "running_mode_converter",
    srcs = ["running_mode_converter.cc"],
    hdrs = ["running_mode_converter.h"],
    deps = [
        ":common",
        "//mediapipe/tasks/cc/audio/core:running_mode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_test(
    name = "running_mode_converter_test",
    srcs = ["running_mode_converter_test.cc"],
    deps = [
        ":common",
        ":running_mode_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/audio/core:running_mode",
        "@com_google_googletest//:gtest_main",
    ],
)
