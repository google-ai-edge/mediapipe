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

IMAGE_CLASSIFIER_DEPS = [
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "//mediapipe/framework/formats:image",
    "//mediapipe/framework/formats:image_frame",
    "//mediapipe/tasks/c/components/containers:classification_result",
    "//mediapipe/tasks/c/components/containers:classification_result_converter",
    "//mediapipe/tasks/c/components/processors:classifier_options",
    "//mediapipe/tasks/c/components/processors:classifier_options_converter",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/c/vision/core:common",
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/core:image_frame_util",
    "//mediapipe/tasks/c/vision/core:image_processing_options",
    "//mediapipe/tasks/c/vision/core:image_processing_options_converter",
    "//mediapipe/tasks/cc/vision/core:image_processing_options",
    "//mediapipe/tasks/cc/vision/core:running_mode",
    "//mediapipe/tasks/cc/vision/image_classifier",
    "//mediapipe/tasks/cc/vision/utils:image_utils",
]

cc_library(
    name = "image_classifier_lib",
    srcs = ["image_classifier.cc"],
    hdrs = ["image_classifier.h"],
    visibility = ["//visibility:public"],
    deps = IMAGE_CLASSIFIER_DEPS,
)

cc_library(
    name = "image_classifier_c_lib",
    srcs = ["image_classifier.cc"],
    hdrs = ["image_classifier.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = IMAGE_CLASSIFIER_DEPS,
    alwayslink = 1,
)

cc_test(
    name = "image_classifier_test",
    srcs = ["image_classifier_test.cc"],
    data = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    linkstatic = 1,
    deps = [
        ":image_classifier_lib",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/core:mp_status",
        "//mediapipe/tasks/c/vision/core:common",
        "//mediapipe/tasks/c/vision/core:image",
        "//mediapipe/tasks/c/vision/core:image_processing_options",
        "//mediapipe/tasks/c/vision/core:image_test_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

# bazel build -c opt --linkopt -s --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/image_classifier:libimage_classifier.so
cc_binary(
    name = "libimage_classifier.so",
    linkopts = [
        "-Wl,-soname=libimage_classifier.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":image_classifier_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/image_classifier:libimage_classifier.dylib
cc_binary(
    name = "libimage_classifier.dylib",
    linkopts = [
        "-Wl,-install_name,libimage_classifier.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":image_classifier_lib"],
)
