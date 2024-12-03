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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_library(
    name = "base_options",
    hdrs = ["base_options.h"],
)

cc_library(
    name = "base_options_converter",
    srcs = ["base_options_converter.cc"],
    hdrs = ["base_options_converter.h"],
    deps = [
        ":base_options",
        "//mediapipe/tasks/cc/core:base_options",
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
