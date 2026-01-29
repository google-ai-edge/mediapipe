# Copyright 2026 The MediaPipe Authors.
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

licenses(["notice"])

cc_library(
    name = "holistic_landmarker_result",
    hdrs = ["holistic_landmarker_result.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/c/vision/core:image",
    ],
)

cc_library(
    name = "holistic_landmarker_result_converter",
    srcs = ["holistic_landmarker_result_converter.cc"],
    hdrs = ["holistic_landmarker_result_converter.h"],
    deps = [
        ":holistic_landmarker_result",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:category_converter",
        "//mediapipe/tasks/c/components/containers:landmark_converter",
        "//mediapipe/tasks/c/vision/core:image_frame_util",
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/holistic_landmarker:holistic_landmarker_result",
    ],
)

HOLISTIC_LANDMARKER_DEPS = [
    ":holistic_landmarker_result",
    ":holistic_landmarker_result_converter",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/log:absl_check",
    "//mediapipe/framework/formats:image",
    "//mediapipe/tasks/c/core:base_options",
    "//mediapipe/tasks/c/core:base_options_converter",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/c/core:common",
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/core:image_frame_util",
    "//mediapipe/tasks/c/vision/core:image_processing_options",
    "//mediapipe/tasks/c/vision/core:image_processing_options_converter",
    "//mediapipe/tasks/cc/vision/core:image_processing_options",
    "//mediapipe/tasks/cc/vision/core:running_mode",
    "//mediapipe/tasks/cc/vision/holistic_landmarker",
    "//mediapipe/tasks/cc/vision/holistic_landmarker:holistic_landmarker_result",
    "//mediapipe/tasks/cc/vision/utils:image_utils",
]

cc_library(
    name = "holistic_landmarker_lib",
    srcs = ["holistic_landmarker.cc"],
    hdrs = ["holistic_landmarker.h"],
    visibility = ["//visibility:public"],
    deps = HOLISTIC_LANDMARKER_DEPS,
)

cc_library(
    name = "holistic_landmarker_c_lib",
    srcs = ["holistic_landmarker.cc"],
    hdrs = ["holistic_landmarker.h"],
    visibility = ["//mediapipe/tasks/c:__subpackages__"],
    deps = HOLISTIC_LANDMARKER_DEPS,
    alwayslink = 1,
)
