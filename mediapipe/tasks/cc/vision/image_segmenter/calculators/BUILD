# Copyright 2022 The MediaPipe Authors.
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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_proto_library(
    name = "tensors_to_segmentation_calculator_proto",
    srcs = ["tensors_to_segmentation_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework/formats:image_format_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_proto",
        "//mediapipe/util:label_map_proto",
    ],
)

config_setting(
    name = "force_gpu_segmentation",
    values = {
        "define": "mediapipe_force_gpu_segmentation=1",
    },
)

cc_library(
    name = "tensors_to_segmentation_calculator",
    srcs = ["tensors_to_segmentation_calculator.cc"],
    deps = [
        ":tensors_to_segmentation_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "//mediapipe/util:label_map_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:emscripten": [
            ":segmentation_postprocessor_gl",
        ],
        "//mediapipe:android": [
            ":segmentation_postprocessor_gl",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "segmentation_postprocessor_gl",
    srcs = ["segmentation_postprocessor_gl.cc"],
    hdrs = ["segmentation_postprocessor_gl.h"],
    tags = ["nomac"],
    deps = [
        ":tensors_to_segmentation_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_cc_proto",
        "//mediapipe/tasks/cc/vision/utils:image_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ] + select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "ssbo_to_texture_converter",
        ],
    }),
)

cc_library(
    name = "ssbo_to_texture_converter",
    srcs = ["ssbo_to_texture_converter.cc"],
    hdrs = ["ssbo_to_texture_converter.h"],
    tags = [
        "nomac",
        "notap",
    ],
    deps = [
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/gpu:gl_base",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_program",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_shader",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl:gl_texture",
        "@org_tensorflow//tensorflow/lite/delegates/gpu/gl/converters:util",
    ],
)

cc_test(
    name = "tensors_to_segmentation_calculator_test",
    srcs = ["tensors_to_segmentation_calculator_test.cc"],
    deps = [
        ":tensors_to_segmentation_calculator",
        ":tensors_to_segmentation_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)
