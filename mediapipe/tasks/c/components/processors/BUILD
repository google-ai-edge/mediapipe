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
    name = "classifier_options",
    hdrs = ["classifier_options.h"],
)

cc_library(
    name = "classifier_options_converter",
    srcs = ["classifier_options_converter.cc"],
    hdrs = ["classifier_options_converter.h"],
    deps = [
        ":classifier_options",
        "//mediapipe/tasks/cc/components/processors:classifier_options",
    ],
)

cc_test(
    name = "classifier_options_converter_test",
    srcs = ["classifier_options_converter_test.cc"],
    deps = [
        ":classifier_options",
        ":classifier_options_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/processors:classifier_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "embedder_options",
    hdrs = ["embedder_options.h"],
)

cc_library(
    name = "embedder_options_converter",
    srcs = ["embedder_options_converter.cc"],
    hdrs = ["embedder_options_converter.h"],
    deps = [
        ":embedder_options",
        "//mediapipe/tasks/cc/components/processors:embedder_options",
    ],
)

cc_test(
    name = "embedder_options_converter_test",
    srcs = ["embedder_options_converter_test.cc"],
    deps = [
        ":embedder_options",
        ":embedder_options_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/components/processors:embedder_options",
        "@com_google_googletest//:gtest_main",
    ],
)
