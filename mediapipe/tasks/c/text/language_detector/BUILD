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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

cc_library(
    name = "language_detector_lib",
    srcs = ["language_detector.cc"],
    hdrs = ["language_detector.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":language_detector_result_converter",
        "//mediapipe/tasks/c/components/processors:classifier_options",
        "//mediapipe/tasks/c/components/processors:classifier_options_converter",
        "//mediapipe/tasks/c/core:base_options",
        "//mediapipe/tasks/c/core:base_options_converter",
        "//mediapipe/tasks/cc/text/language_detector",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text/language_detector:liblanguage_detector.so
cc_binary(
    name = "liblanguage_detector.so",
    linkopts = [
        "-Wl,-soname=liblanguage_detector.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":language_detector_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/text/language_detector:liblanguage_detector.dylib
cc_binary(
    name = "liblanguage_detector.dylib",
    linkopts = [
        "-Wl,-install_name,liblanguage_detector.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":language_detector_lib"],
)

cc_library(
    name = "language_detector",
    hdrs = ["language_detector.h"],
    deps = [
        "//mediapipe/tasks/c/components/processors:classifier_options",
        "//mediapipe/tasks/c/core:base_options",
    ],
)

cc_library(
    name = "language_detector_result_converter",
    srcs = ["language_detector_result_converter.cc"],
    hdrs = ["language_detector_result_converter.h"],
    deps = [
        ":language_detector",
        "//mediapipe/tasks/cc/text/language_detector",
    ],
)

cc_test(
    name = "language_detector_result_converter_test",
    srcs = ["language_detector_result_converter_test.cc"],
    linkstatic = 1,
    deps = [
        ":language_detector",
        ":language_detector_result_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/cc/text/language_detector",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "language_detector_test",
    srcs = ["language_detector_test.cc"],
    data = ["//mediapipe/tasks/testdata/text:language_detector"],
    linkstatic = 1,
    deps = [
        ":language_detector_lib",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
