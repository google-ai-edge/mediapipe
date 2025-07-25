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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "opencv_image_encoder_calculator_proto",
    srcs = ["opencv_image_encoder_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "scale_image_calculator_proto",
    srcs = ["scale_image_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/framework/formats:image_format_proto",
    ],
)

mediapipe_proto_library(
    name = "set_alpha_calculator_proto",
    srcs = ["set_alpha_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "image_cropping_calculator_proto",
    srcs = ["image_cropping_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "bilateral_filter_calculator_proto",
    srcs = ["bilateral_filter_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "recolor_calculator_proto",
    srcs = ["recolor_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
    ],
)

mediapipe_proto_library(
    name = "segmentation_smoothing_calculator_proto",
    srcs = ["segmentation_smoothing_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "color_convert_calculator",
    srcs = ["color_convert_calculator.cc"],
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
        "@com_google_absl//absl/log:absl_check",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_encoded_image_to_image_frame_calculator",
    srcs = ["opencv_encoded_image_to_image_frame_calculator.cc"],
    deps = [
        ":opencv_encoded_image_to_image_frame_calculator_cc_proto",
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
    deps = [
        ":opencv_image_encoder_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_check",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_put_text_calculator",
    srcs = ["opencv_put_text_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "set_alpha_calculator",
    srcs = ["set_alpha_calculator.cc"],
    deps = [
        ":set_alpha_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_log",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:shader_util",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "set_alpha_calculator_test",
    srcs = ["set_alpha_calculator_test.cc"],
    deps = [
        ":set_alpha_calculator",
        ":set_alpha_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bilateral_filter_calculator",
    srcs = ["bilateral_filter_calculator.cc"],
    deps = [
        ":bilateral_filter_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:shader_util",
        ],
    }),
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "rotation_mode_proto",
    srcs = ["rotation_mode.proto"],
)

mediapipe_proto_library(
    name = "image_transformation_calculator_proto",
    srcs = ["image_transformation_calculator.proto"],
    deps = [
        ":rotation_mode_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:scale_mode_proto",
    ],
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
    deps = [
        ":image_transformation_calculator_cc_proto",
        ":rotation_mode_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:scale_mode_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_base_hdr",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:shader_util",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "image_transformation_calculator_test",
    srcs = ["image_transformation_calculator_test.cc"],
    data = ["//mediapipe/calculators/image/testdata:test_images"],
    tags = [
        "desktop_only_test",
        "noasan",  # TODO: Remove noasan tag when SwiftShader supports sanitizers],
    ],
    deps = [
        ":image_transformation_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "//mediapipe/gpu:multi_pool",
        "//third_party:opencv",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "image_cropping_calculator",
    srcs = ["image_cropping_calculator.cc"],
    hdrs = ["image_cropping_calculator.h"],
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
        "@com_google_absl//absl/log:absl_log",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "image_cropping_calculator_test",
    srcs = ["image_cropping_calculator_test.cc"],
    tags = [
        "noasan",  # TODO: Remove noasan tag when SwiftShader supports sanitizers
        "not_run:arm",
    ],
    deps = [
        ":image_cropping_calculator",
        ":image_cropping_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:calculator_state",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:tag_map",
        "//mediapipe/framework/tool:tag_map_helper",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
    ],
)

cc_library(
    name = "luminance_calculator",
    srcs = ["luminance_calculator.cc"],
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
    deps = [
        ":recolor_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:color_cc_proto",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:shader_util",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "scale_image_utils",
    srcs = ["scale_image_utils.cc"],
    hdrs = ["scale_image_utils.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "scale_image_calculator",
    srcs = ["scale_image_calculator.cc"],
    deps = [
        ":scale_image_calculator_cc_proto",
        ":scale_image_utils",
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
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@libyuv",
    ],
    alwayslink = 1,
)

cc_test(
    name = "scale_image_calculator_test",
    srcs = ["scale_image_calculator_test.cc"],
    deps = [
        ":scale_image_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/status",
    ],
)

mediapipe_proto_library(
    name = "image_clone_calculator_proto",
    srcs = ["image_clone_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "image_clone_calculator",
    srcs = ["image_clone_calculator.cc"],
    deps = [
        ":image_clone_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "image_properties_calculator",
    srcs = ["image_properties_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gpu_buffer",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "image_properties_calculator_test",
    srcs = ["image_properties_calculator_test.cc"],
    data = ["//mediapipe/calculators/image/testdata:test_images"],
    deps = [
        ":image_properties_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:test_util",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
    ],
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

mediapipe_proto_library(
    name = "mask_overlay_calculator_proto",
    srcs = ["mask_overlay_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "opencv_encoded_image_to_image_frame_calculator_proto",
    srcs = ["opencv_encoded_image_to_image_frame_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "feature_detector_calculator_proto",
    srcs = ["feature_detector_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "mask_overlay_calculator",
    srcs = ["mask_overlay_calculator.cc"],
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

cc_test(
    name = "mask_overlay_calculator_test",
    srcs = ["mask_overlay_calculator_test.cc"],
    data = ["//mediapipe/calculators/image/testdata:test_images"],
    tags = [
        "angle",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":mask_overlay_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:test_util",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "//third_party:opencv",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "feature_detector_calculator",
    srcs = ["feature_detector_calculator.cc"],
    deps = [
        ":feature_detector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_features2d",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/framework/tool:options_util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "image_file_properties_calculator",
    srcs = ["image_file_properties_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_file_properties_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@easyexif",
    ],
    alwayslink = 1,
)

cc_test(
    name = "image_file_properties_calculator_test",
    srcs = ["image_file_properties_calculator_test.cc"],
    data = ["//mediapipe/calculators/image/testdata:test_images"],
    deps = [
        ":image_file_properties_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_file_properties_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_library(
    name = "segmentation_smoothing_calculator",
    srcs = ["segmentation_smoothing_calculator.cc"],
    deps = [
        ":segmentation_smoothing_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_log",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_quad_renderer",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:shader_util",
        ],
    }) + select({
        "//mediapipe/framework/port:disable_opencv": [],
        "//conditions:default": [
            "//mediapipe/framework/formats:image_frame_opencv",
            "//mediapipe/framework/formats:image_opencv",
            "//mediapipe/framework/port:opencv_core",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "segmentation_smoothing_calculator_test",
    srcs = ["segmentation_smoothing_calculator_test.cc"],
    tags = [
        "noasan",  # TODO: Remove noasan tag when SwiftShader supports sanitizers
        "not_run:arm",
    ],
    deps = [
        ":image_clone_calculator",
        ":image_clone_calculator_cc_proto",
        ":segmentation_smoothing_calculator",
        ":segmentation_smoothing_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_opencv",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "affine_transformation",
    hdrs = ["affine_transformation.h"],
    deps = ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "affine_transformation_runner_gl",
    srcs = ["affine_transformation_runner_gl.cc"],
    hdrs = ["affine_transformation_runner_gl.h"],
    deps = [
        ":affine_transformation",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gl_simple_shaders",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_origin_cc_proto",
        "//mediapipe/gpu:gpu_origin_utils",
        "//mediapipe/gpu:shader_util",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@eigen//:eigen3",
    ],
)

cc_library(
    name = "affine_transformation_runner_opencv",
    srcs = ["affine_transformation_runner_opencv.cc"],
    hdrs = ["affine_transformation_runner_opencv.h"],
    deps = [
        ":affine_transformation",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status:statusor",
        "@eigen//:eigen3",
    ],
)

mediapipe_proto_library(
    name = "warp_affine_calculator_proto",
    srcs = ["warp_affine_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/gpu:gpu_origin_proto",
    ],
)

cc_library(
    name = "warp_affine_calculator",
    srcs = ["warp_affine_calculator.cc"],
    hdrs = ["warp_affine_calculator.h"],
    deps = [
        ":affine_transformation",
        ":warp_affine_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/api3:calculator",
        "//mediapipe/framework/api3:calculator_context",
        "//mediapipe/framework/api3:calculator_contract",
        "//mediapipe/framework/api3:contract",
        "//mediapipe/framework/api3:node",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            ":affine_transformation_runner_gl",
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:gpu_service",
        ],
    }) + select({
        "//mediapipe/framework/port:disable_opencv": [],
        "//conditions:default": [":affine_transformation_runner_opencv"],
    }),
    alwayslink = 1,
)

cc_test(
    name = "warp_affine_calculator_test",
    srcs = ["warp_affine_calculator_test.cc"],
    data = [
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/input.jpg",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/large_sub_rect.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/large_sub_rect_border_zero.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/large_sub_rect_keep_aspect.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/large_sub_rect_keep_aspect_border_zero.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/large_sub_rect_keep_aspect_with_rotation.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/large_sub_rect_keep_aspect_with_rotation_border_zero.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/medium_sub_rect_keep_aspect.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/medium_sub_rect_keep_aspect_border_zero.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/medium_sub_rect_keep_aspect_with_rotation.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/medium_sub_rect_keep_aspect_with_rotation_border_zero.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/medium_sub_rect_with_rotation.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/medium_sub_rect_with_rotation_border_zero.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/medium_sub_rect_with_rotation_border_zero_interp_cubic.png",
        "//mediapipe/calculators/tensor:testdata/image_to_tensor/noop_except_range.png",
    ],
    tags = [
        "desktop_only_test",
        "noasan",
        "not_run:arm",  # TODO: Remove noasan tag when SwiftShader supports sanitizers
    ],
    deps = [
        ":affine_transformation",
        ":image_transformation_calculator",
        ":warp_affine_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_converter",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "yuv_to_image_calculator",
    srcs = ["yuv_to_image_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:yuv_image",
        "//third_party/libyuv",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)
