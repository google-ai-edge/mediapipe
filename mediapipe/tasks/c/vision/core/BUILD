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
    name = "common",
    hdrs = ["common.h"],
)

cc_library(
    name = "image_processing_options",
    hdrs = ["image_processing_options.h"],
    deps = [
        "//mediapipe/tasks/c/components/containers:rect",
    ],
)

cc_library(
    name = "image_processing_options_converter",
    srcs = ["image_processing_options_converter.cc"],
    hdrs = ["image_processing_options_converter.h"],
    deps = [
        ":image_processing_options",
        "//mediapipe/tasks/cc/components/containers:rect",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
    ],
)

cc_test(
    name = "image_processing_options_converter_test",
    srcs = ["image_processing_options_converter_test.cc"],
    deps = [
        ":common",
        ":image_processing_options",
        ":image_processing_options_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/containers:rect",
        "//mediapipe/tasks/cc/vision/core:image_processing_options",
        "@com_google_googletest//:gtest_main",
    ],
)
