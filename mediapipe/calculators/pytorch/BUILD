# Copyright 2020 The MediaPipe Authors.
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

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

mediapipe_proto_library(
    name = "pytorch_converter_calculator_proto",
    srcs = ["pytorch_converter_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "pytorch_inference_calculator_proto",
    srcs = ["pytorch_inference_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "pytorch_tensors_to_classification_calculator_proto",
    srcs = ["pytorch_tensors_to_classification_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "pytorch_converter_calculator",
    srcs = ["pytorch_converter_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":pytorch_converter_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//third_party:libtorch",
    ] + select({
        "//mediapipe:ios": [
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/objc:util",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "pytorch_inference_calculator",
    srcs = ["pytorch_inference_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":pytorch_inference_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/util:resource_util",
        "//third_party:libtorch",
    ],
    alwayslink = 1,
)

cc_library(
    name = "pytorch_tensors_to_classification_calculator",
    srcs = ["pytorch_tensors_to_classification_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":pytorch_tensors_to_classification_calculator_cc_proto",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:resource_util",
        "//third_party:libtorch",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:apple": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:macos": [
            "//mediapipe/framework/port:file_helpers",
        ],
        "//conditions:default": [
            "//mediapipe/framework/port:file_helpers",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "pytorch_converter_calculator_test",
    srcs = ["pytorch_converter_calculator_test.cc"],
    deps = [
        ":pytorch_converter_calculator",
        ":pytorch_converter_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        # "//mediapipe/framework:calculator_runner",
        # "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest_main",
        # "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        # "//mediapipe/framework/tool:validate_type",
    ],
)
