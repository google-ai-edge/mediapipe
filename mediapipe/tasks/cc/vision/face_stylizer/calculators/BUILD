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

load("@bazel_skylib//lib:selects.bzl", "selects")
load("//mediapipe/framework:more_selects.bzl", "more_selects")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//mediapipe/tasks:internal"])

config_setting(
    name = "disable_gpu",
    define_values = {
        "MEDIAPIPE_DISABLE_GPU": "1",
    },
    visibility = ["//visibility:public"],
)

more_selects.config_setting_negation(
    name = "not_disable_gpu",
    negate = ":disable_gpu",
)

selects.config_setting_group(
    name = "platform_apple_with_gpu",
    match_all = [
        ":not_disable_gpu",
        "//mediapipe:apple",
    ],
)

selects.config_setting_group(
    name = "platform_apple_without_gpu",
    match_all = [
        ":disable_gpu",
        "//mediapipe:apple",
    ],
)

mediapipe_proto_library(
    name = "tensors_to_image_calculator_proto",
    srcs = ["tensors_to_image_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:gpu_origin_proto",
    ],
)

cc_library(
    name = "tensors_to_image_calculator",
    srcs = ["tensors_to_image_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],  # allow depending on tensor_to_image_calculator_gpu_deps
    linkopts = select({
        "//mediapipe:apple": [
            "-framework CoreVideo",
            "-framework MetalKit",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":tensors_to_image_calculator_cc_proto",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/api2:builder",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "//mediapipe/gpu:gpu_origin_cc_proto",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": ["tensor_to_image_calculator_gpu_deps"],
    }) + select({
        "//mediapipe:apple": [
            "//third_party/apple_frameworks:CoreVideo",
            "//third_party/apple_frameworks:MetalKit",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tensor_to_image_calculator_gpu_deps",
    visibility = ["//visibility:private"],
    deps = selects.with_or({
        "//mediapipe:android": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:util",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_texture",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl/converters:util",
        ],
        ":platform_apple_with_gpu": [
            "//mediapipe/gpu:MPPMetalHelper",
            "//mediapipe/gpu:MPPMetalUtil",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gpu_buffer",
        ],
        # This setting is needed to allow bazel to build all targets on Mac with GPU disabled
        ":platform_apple_without_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gpu_buffer",
            "@org_tensorflow//tensorflow/lite/delegates/gpu:gl_delegate",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/common:util",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_texture",
            "@org_tensorflow//tensorflow/lite/delegates/gpu/gl/converters:util",
        ],
    }),
)

cc_library(
    name = "strip_rotation_calculator",
    srcs = ["strip_rotation_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:rect_cc_proto",
    ],
    alwayslink = 1,
)
