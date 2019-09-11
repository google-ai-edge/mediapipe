# Copyright 2019 The MediaPipe Authors.
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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

exports_files(["LICENSE"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

proto_library(
    name = "opencv_image_encoder_calculator_proto",
    srcs = ["opencv_image_encoder_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "scale_image_calculator_proto",
    srcs = ["scale_image_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework/formats:image_format_proto",
    ],
)

proto_library(
    name = "set_alpha_calculator_proto",
    srcs = ["set_alpha_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "image_cropping_calculator_proto",
    srcs = ["image_cropping_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "bilateral_filter_calculator_proto",
    srcs = ["bilateral_filter_calculator.proto"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "recolor_calculator_proto",
    srcs = ["recolor_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "opencv_image_encoder_calculator_cc_proto",
    srcs = ["opencv_image_encoder_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":opencv_image_encoder_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "mask_overlay_calculator_cc_proto",
    srcs = ["mask_overlay_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":mask_overlay_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "scale_image_calculator_cc_proto",
    srcs = ["scale_image_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":scale_image_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "set_alpha_calculator_cc_proto",
    srcs = ["set_alpha_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":set_alpha_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "image_cropping_calculator_cc_proto",
    srcs = ["image_cropping_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":image_cropping_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "bilateral_filter_calculator_cc_proto",
    srcs = ["bilateral_filter_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = [
        "//visibility:public",
    ],
    deps = [":bilateral_filter_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "recolor_calculator_cc_proto",
    srcs = ["recolor_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":recolor_calculator_proto"],
)

cc_library(
    name = "color_convert_calculator",
    srcs = ["color_convert_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_encoded_image_to_image_frame_calculator",
    srcs = ["opencv_encoded_image_to_image_frame_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_image_encoder_calculator",
    srcs = ["opencv_image_encoder_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":opencv_image_encoder_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_put_text_calculator",
    srcs = ["opencv_put_text_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "set_alpha_calculator",
    srcs = ["set_alpha_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":set_alpha_calculator_cc_proto",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
        "//mediapipe:ios": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "bilateral_filter_calculator",
    srcs = ["bilateral_filter_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":bilateral_filter_calculator_cc_proto",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

proto_library(
    name = "image_transformation_calculator_proto",
    srcs = ["image_transformation_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:scale_mode_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "image_transformation_calculator_cc_proto",
    srcs = ["image_transformation_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/gpu:scale_mode_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":image_transformation_calculator_proto"],
)

cc_library(
    name = "image_transformation_calculator",
    srcs = ["image_transformation_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//mediapipe:apple": [
            "-framework CoreVideo",
            "-framework MetalKit",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":image_transformation_calculator_cc_proto",
        "//mediapipe/gpu:scale_mode_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ] + selects.with_or({
        ("//mediapipe:android", "//mediapipe:ios"): [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:shader_util",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "image_cropping_calculator",
    srcs = ["image_cropping_calculator.cc"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//mediapipe:apple": [
            "-framework CoreVideo",
            "-framework MetalKit",
        ],
        "//conditions:default": [],
    }),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":image_cropping_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ] + selects.with_or({
        ("//mediapipe:android", "//mediapipe:ios"): [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:shader_util",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "luminance_calculator",
    srcs = ["luminance_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_simple_calculator",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sobel_edges_calculator",
    srcs = ["sobel_edges_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_simple_calculator",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "recolor_calculator",
    srcs = ["recolor_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":recolor_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:color_cc_proto",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
        "//mediapipe:ios": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "scale_image_utils",
    srcs = ["scale_image_utils.cc"],
    hdrs = ["scale_image_utils.h"],
    visibility = [
        "//mediapipe:__subpackages__",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "scale_image_calculator",
    srcs = ["scale_image_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":scale_image_utils",
        "//mediapipe/calculators/image:scale_image_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/formats:yuv_image",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:image_resizer",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:image_frame_util",
        "@com_google_absl//absl/strings",
        "@libyuv",
    ],
    alwayslink = 1,
)

cc_library(
    name = "image_properties_calculator",
    srcs = ["image_properties_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ] + selects.with_or({
        ("//mediapipe:android", "//mediapipe:ios"): [
            "//mediapipe/gpu:gpu_buffer",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_test(
    name = "opencv_encoded_image_to_image_frame_calculator_test",
    srcs = ["opencv_encoded_image_to_image_frame_calculator_test.cc"],
    data = ["//mediapipe/calculators/image/testdata:test_images"],
    deps = [
        ":opencv_encoded_image_to_image_frame_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

cc_test(
    name = "opencv_image_encoder_calculator_test",
    srcs = ["opencv_image_encoder_calculator_test.cc"],
    data = ["//mediapipe/calculators/image/testdata:test_images"],
    deps = [
        ":opencv_image_encoder_calculator",
        ":opencv_image_encoder_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

cc_test(
    name = "scale_image_utils_test",
    srcs = ["scale_image_utils_test.cc"],
    deps = [
        ":scale_image_utils",
        "//mediapipe/framework/port:gtest_main",
    ],
)

proto_library(
    name = "mask_overlay_calculator_proto",
    srcs = ["mask_overlay_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

cc_library(
    name = "mask_overlay_calculator",
    srcs = ["mask_overlay_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":mask_overlay_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:shader_util",
    ],
    alwayslink = 1,
)
