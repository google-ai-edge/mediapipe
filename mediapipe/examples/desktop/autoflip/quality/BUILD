load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

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

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

proto_library(
    name = "cropping_proto",
    srcs = ["cropping.proto"],
    deps = [
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "cropping_cc_proto",
    srcs = ["cropping.proto"],
    cc_deps = ["//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto"],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":cropping_proto"],
)

proto_library(
    name = "focus_point_proto",
    srcs = ["focus_point.proto"],
)

mediapipe_cc_proto_library(
    name = "focus_point_cc_proto",
    srcs = ["focus_point.proto"],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":focus_point_proto"],
)

cc_library(
    name = "frame_crop_region_computer",
    srcs = ["frame_crop_region_computer.cc"],
    hdrs = ["frame_crop_region_computer.h"],
    deps = [
        ":cropping_cc_proto",
        ":utils",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
)

cc_library(
    name = "piecewise_linear_function",
    srcs = ["piecewise_linear_function.cc"],
    hdrs = ["piecewise_linear_function.h"],
    deps = [
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "padding_effect_generator",
    srcs = ["padding_effect_generator.cc"],
    hdrs = ["padding_effect_generator.h"],
    deps = [
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "scene_camera_motion_analyzer",
    srcs = ["scene_camera_motion_analyzer.cc"],
    hdrs = ["scene_camera_motion_analyzer.h"],
    deps = [
        ":cropping_cc_proto",
        ":focus_point_cc_proto",
        ":math_utils",
        ":piecewise_linear_function",
        ":utils",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "scene_cropping_viz",
    srcs = ["scene_cropping_viz.cc"],
    hdrs = ["scene_cropping_viz.h"],
    deps = [
        ":cropping_cc_proto",
        ":focus_point_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "polynomial_regression_path_solver",
    srcs = ["polynomial_regression_path_solver.cc"],
    hdrs = ["polynomial_regression_path_solver.h"],
    deps = [
        ":focus_point_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@ceres_solver//:ceres",
    ],
)

cc_library(
    name = "scene_cropper",
    srcs = ["scene_cropper.cc"],
    hdrs = ["scene_cropper.h"],
    deps = [
        ":cropping_cc_proto",
        ":focus_point_cc_proto",
        ":polynomial_regression_path_solver",
        ":utils",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":cropping_cc_proto",
        ":math_utils",
        ":piecewise_linear_function",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "frame_crop_region_computer_test",
    srcs = ["frame_crop_region_computer_test.cc"],
    deps = [
        ":cropping_cc_proto",
        ":frame_crop_region_computer",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "piecewise_linear_function_test",
    srcs = ["piecewise_linear_function_test.cc"],
    deps = [
        ":piecewise_linear_function",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
    ],
)

cc_test(
    name = "scene_camera_motion_analyzer_test",
    srcs = ["scene_camera_motion_analyzer_test.cc"],
    data = [
        "//mediapipe/examples/desktop/autoflip/quality/testdata:camera_motion_tracking_scene_frame_results.csv",
    ],
    deps = [
        ":focus_point_cc_proto",
        ":piecewise_linear_function",
        ":scene_camera_motion_analyzer",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "padding_effect_generator_test",
    srcs = ["padding_effect_generator_test.cc"],
    data = [
        "//mediapipe/examples/desktop/autoflip/quality/testdata:google.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_0.3.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_0.3_solid_background.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_0.6.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_0.6_solid_background.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_1.6.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_1.6_solid_background.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_1.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_1_solid_background.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_2.5.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_2.5_solid_background.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_3.4.jpg",
        "//mediapipe/examples/desktop/autoflip/quality/testdata:result_3.4_solid_background.jpg",
    ],
    deps = [
        ":padding_effect_generator",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "polynomial_regression_path_solver_test",
    srcs = ["polynomial_regression_path_solver_test.cc"],
    deps = [
        ":focus_point_cc_proto",
        ":polynomial_regression_path_solver",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:status",
    ],
)

cc_test(
    name = "scene_cropper_test",
    size = "small",
    timeout = "short",
    srcs = ["scene_cropper_test.cc"],
    deps = [
        ":focus_point_cc_proto",
        ":scene_cropper",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":cropping_cc_proto",
        ":utils",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_core",
    ],
)

proto_library(
    name = "visual_scorer_proto",
    srcs = ["visual_scorer.proto"],
)

mediapipe_cc_proto_library(
    name = "visual_scorer_cc_proto",
    srcs = ["visual_scorer.proto"],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":visual_scorer_proto"],
)

cc_library(
    name = "visual_scorer",
    srcs = ["visual_scorer.cc"],
    hdrs = ["visual_scorer.h"],
    deps = [
        ":visual_scorer_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

cc_test(
    name = "visual_scorer_test",
    srcs = [
        "visual_scorer_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":visual_scorer",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)
