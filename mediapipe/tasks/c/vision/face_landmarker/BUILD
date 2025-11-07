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
    name = "face_landmarker_result",
    hdrs = ["face_landmarker_result.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/c/components/containers:matrix",
    ],
)

cc_library(
    name = "face_landmarker_result_converter",
    srcs = ["face_landmarker_result_converter.cc"],
    hdrs = ["face_landmarker_result_converter.h"],
    deps = [
        ":face_landmarker_result",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:category_converter",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/c/components/containers:landmark_converter",
        "//mediapipe/tasks/c/components/containers:matrix",
        "//mediapipe/tasks/c/components/containers:matrix_converter",
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_result",
    ],
)

cc_test(
    name = "face_landmarker_result_converter_test",
    srcs = ["face_landmarker_result_converter_test.cc"],
    linkstatic = 1,
    deps = [
        ":face_landmarker_result",
        ":face_landmarker_result_converter",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_result",
        "@com_google_googletest//:gtest_main",
        "@eigen//:eigen3",
    ],
)

FACE_LANDMARKER_DEPS = [
    ":face_landmarker_result",
    ":face_landmarker_result_converter",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "//mediapipe/framework/formats:image",
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
    "//mediapipe/tasks/cc/vision/face_landmarker",
    "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_result",
    "//mediapipe/tasks/cc/vision/utils:image_utils",
]

cc_library(
    name = "face_landmarker_lib",
    srcs = ["face_landmarker.cc"],
    hdrs = ["face_landmarker.h"],
    visibility = ["//visibility:public"],
    deps = FACE_LANDMARKER_DEPS,
)

cc_library(
    name = "face_landmarker_c_lib",
    srcs = ["face_landmarker.cc"],
    hdrs = ["face_landmarker.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = FACE_LANDMARKER_DEPS,
    alwayslink = 1,
)

cc_test(
    name = "face_landmarker_test",
    srcs = ["face_landmarker_test.cc"],
    data = [
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    linkstatic = 1,
    deps = [
        ":face_landmarker_lib",
        ":face_landmarker_result",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/components/containers:landmark",
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
# //mediapipe/tasks/c/vision/face_landmarker:libface_landmarker.so
cc_binary(
    name = "libface_landmarker.so",
    linkopts = [
        "-Wl,-soname=libface_landmarker.so",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":face_landmarker_lib"],
)

# bazel build --config darwin_arm64 -c opt --strip always --define MEDIAPIPE_DISABLE_GPU=1 \
# //mediapipe/tasks/c/vision/face_landmarker:libface_landmarker.dylib
cc_binary(
    name = "libface_landmarker.dylib",
    linkopts = [
        "-Wl,-install_name,libface_landmarker.dylib",
        "-fvisibility=hidden",
    ],
    linkshared = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [":face_landmarker_lib"],
)
