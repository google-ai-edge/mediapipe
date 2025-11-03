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
    srcs = ["common.cc"],
    hdrs = ["common.h"],
)

cc_library(
    name = "image_hdr",
    hdrs = ["image.h"],
    visibility = ["//visibility:private"],
    deps = ["//mediapipe/tasks/c/core:mp_status"],
)

IMAGE_DEPS = [
    ":common",
    ":image_frame_util",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings:str_format",
    "//mediapipe/framework/formats:image",
    "//mediapipe/framework/formats:image_frame",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "@stblib//:stb_image",
]

cc_library(
    name = "image",
    srcs = ["image.cc"],
    hdrs = ["image.h"],
    deps = IMAGE_DEPS,
)

cc_library(
    name = "image_c_lib",
    hdrs = ["image.h"],
    deps = IMAGE_DEPS + [":image"],
    alwayslink = 1,
)

cc_test(
    name = "image_test",
    srcs = ["image_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
    ],
    deps = [
        ":image",
        ":image_frame_util",
        ":image_test_util",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/tasks/c/core:mp_status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "image_frame_util",
    hdrs = ["image_frame_util.h"],
    deps = [
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
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

cc_library(
    name = "image_test_util",
    testonly = True,
    srcs = ["image_test_util.cc"],
    hdrs = ["image_test_util.h"],
    deps = [
        ":image_frame_util",
        ":image_hdr",
        "//mediapipe/framework/formats:image",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/tasks/c/core:mp_status",
        "@com_google_absl//absl/log:absl_check",
    ],
)
