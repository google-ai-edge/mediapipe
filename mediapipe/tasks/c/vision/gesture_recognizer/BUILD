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
    name = "gesture_recognizer_result",
    hdrs = ["gesture_recognizer_result.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:landmark",
    ],
)

cc_library(
    name = "gesture_recognizer_result_converter",
    srcs = ["gesture_recognizer_result_converter.cc"],
    hdrs = ["gesture_recognizer_result_converter.h"],
    deps = [
        ":gesture_recognizer_result",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:category_converter",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/c/components/containers:landmark_converter",
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/gesture_recognizer:gesture_recognizer_result",
    ],
)

cc_test(
    name = "gesture_recognizer_result_converter_test",
    srcs = ["gesture_recognizer_result_converter_test.cc"],
    linkstatic = 1,
    deps = [
        ":gesture_recognizer_result",
        ":gesture_recognizer_result_converter",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/gesture_recognizer:gesture_recognizer_result",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gesture_recognizer_lib",
    srcs = ["gesture_recognizer.cc"],
    hdrs = ["gesture_recognizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gesture_recognizer_result",
        ":gesture_recognizer_result_converter",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/tasks/c/components/processors:classifier_options",
        "//mediapipe/tasks/c/components/processors:classifier_options_converter",
        "//mediapipe/tasks/c/core:base_options",
        "//mediapipe/tasks/c/core:base_options_converter",
        "//mediapipe/tasks/c/vision/core:common",
        "//mediapipe/tasks/cc/vision/core:running_mode",
        "//mediapipe/tasks/cc/vision/gesture_recognizer",
        "//mediapipe/tasks/cc/vision/gesture_recognizer:gesture_recognizer_result",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "gesture_recognizer_test",
    srcs = ["gesture_recognizer_test.cc"],
    data = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    linkstatic = 1,
    deps = [
        ":gesture_recognizer_lib",
        ":gesture_recognizer_result",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/c/vision/core:common",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/gesture_recognizer:libgesture_recognizer.so
cc_binary(
    name = "libgesture_recognizer.so",
    linkopts = [
        "-Wl,-soname=libgesture_recognizer.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":gesture_recognizer_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/gesture_recognizer:libgesture_recognizer.dylib
cc_binary(
    name = "libgesture_recognizer.dylib",
    linkopts = [
        "-Wl,-install_name,libgesture_recognizer.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":gesture_recognizer_lib"],
)
