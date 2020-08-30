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

licenses(["notice"])

package(default_visibility = [
    "//mediapipe/examples:__subpackages__",
    "//mediapipe/viz:__subpackages__",
])

cc_library(
    name = "border_detection_calculator",
    srcs = ["border_detection_calculator.cc"],
    deps = [
        ":border_detection_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

proto_library(
    name = "border_detection_calculator_proto",
    srcs = ["border_detection_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "border_detection_calculator_cc_proto",
    srcs = ["border_detection_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":border_detection_calculator_proto"],
)

cc_library(
    name = "content_zooming_calculator",
    srcs = ["content_zooming_calculator.cc"],
    deps = [
        ":content_zooming_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:kinematic_path_solver",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

proto_library(
    name = "content_zooming_calculator_proto",
    srcs = ["content_zooming_calculator.proto"],
    deps = [
        "//mediapipe/examples/desktop/autoflip/quality:kinematic_path_solver_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "content_zooming_calculator_cc_proto",
    srcs = ["content_zooming_calculator.proto"],
    cc_deps = [
        "//mediapipe/examples/desktop/autoflip/quality:kinematic_path_solver_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [
        ":content_zooming_calculator_proto",
    ],
)

cc_test(
    name = "border_detection_calculator_test",
    srcs = [
        "border_detection_calculator_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":border_detection_calculator",
        ":border_detection_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:benchmark",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "content_zooming_calculator_test",
    srcs = [
        "content_zooming_calculator_test.cc",
    ],
    linkstatic = 1,
    deps = [
        ":content_zooming_calculator",
        ":content_zooming_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:kinematic_path_solver",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:benchmark",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "video_filtering_calculator",
    srcs = ["video_filtering_calculator.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    deps = [
        ":video_filtering_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

proto_library(
    name = "video_filtering_calculator_proto",
    srcs = ["video_filtering_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "video_filtering_calculator_cc_proto",
    srcs = ["video_filtering_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":video_filtering_calculator_proto"],
)

cc_test(
    name = "video_filtering_calculator_test",
    srcs = ["video_filtering_calculator_test.cc"],
    deps = [
        ":video_filtering_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "scene_cropping_calculator_proto",
    srcs = ["scene_cropping_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/examples/desktop/autoflip/quality:cropping_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "scene_cropping_calculator_cc_proto",
    srcs = ["scene_cropping_calculator.proto"],
    cc_deps = [
        "//mediapipe/examples/desktop/autoflip/quality:cropping_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":scene_cropping_calculator_proto"],
)

cc_library(
    name = "scene_cropping_calculator",
    srcs = ["scene_cropping_calculator.cc"],
    hdrs = ["scene_cropping_calculator.h"],
    deps = [
        ":scene_cropping_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:cropping_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:focus_point_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:frame_crop_region_computer",
        "//mediapipe/examples/desktop/autoflip/quality:padding_effect_generator",
        "//mediapipe/examples/desktop/autoflip/quality:piecewise_linear_function",
        "//mediapipe/examples/desktop/autoflip/quality:polynomial_regression_path_solver",
        "//mediapipe/examples/desktop/autoflip/quality:scene_camera_motion_analyzer",
        "//mediapipe/examples/desktop/autoflip/quality:scene_cropper",
        "//mediapipe/examples/desktop/autoflip/quality:scene_cropping_viz",
        "//mediapipe/examples/desktop/autoflip/quality:utils",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,  # buildozer: disable=alwayslink-with-hdrs
)

cc_test(
    name = "scene_cropping_calculator_test",
    size = "large",
    timeout = "long",
    srcs = ["scene_cropping_calculator_test.cc"],
    deps = [
        ":scene_cropping_calculator",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
)

cc_library(
    name = "signal_fusing_calculator",
    srcs = ["signal_fusing_calculator.cc"],
    deps = [
        ":signal_fusing_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

proto_library(
    name = "signal_fusing_calculator_proto",
    srcs = ["signal_fusing_calculator.proto"],
    deps = [
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "signal_fusing_calculator_cc_proto",
    srcs = ["signal_fusing_calculator.proto"],
    cc_deps = [
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":signal_fusing_calculator_proto"],
)

cc_test(
    name = "signal_fusing_calculator_test",
    srcs = ["signal_fusing_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":signal_fusing_calculator",
        ":signal_fusing_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "shot_boundary_calculator",
    srcs = ["shot_boundary_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":shot_boundary_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

proto_library(
    name = "shot_boundary_calculator_proto",
    srcs = ["shot_boundary_calculator.proto"],
    deps = [
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "shot_boundary_calculator_cc_proto",
    srcs = ["shot_boundary_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":shot_boundary_calculator_proto"],
)

cc_test(
    name = "shot_boundary_calculator_test",
    srcs = ["shot_boundary_calculator_test.cc"],
    data = ["//mediapipe/examples/desktop/autoflip/calculators/testdata:test_images"],
    linkstatic = 1,
    deps = [
        ":shot_boundary_calculator",
        ":shot_boundary_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "face_to_region_calculator",
    srcs = ["face_to_region_calculator.cc"],
    deps = [
        ":face_to_region_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/examples/desktop/autoflip/quality:visual_scorer",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

proto_library(
    name = "face_to_region_calculator_proto",
    srcs = ["face_to_region_calculator.proto"],
    deps = [
        "//mediapipe/examples/desktop/autoflip/quality:visual_scorer_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "face_to_region_calculator_cc_proto",
    srcs = ["face_to_region_calculator.proto"],
    cc_deps = [
        "//mediapipe/examples/desktop/autoflip/quality:visual_scorer_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":face_to_region_calculator_proto"],
)

cc_test(
    name = "face_to_region_calculator_test",
    srcs = ["face_to_region_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":face_to_region_calculator",
        ":face_to_region_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "localization_to_region_calculator_proto",
    srcs = ["localization_to_region_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "localization_to_region_calculator_cc_proto",
    srcs = ["localization_to_region_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//mediapipe/examples:__subpackages__"],
    deps = [":localization_to_region_calculator_proto"],
)

cc_library(
    name = "localization_to_region_calculator",
    srcs = ["localization_to_region_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":localization_to_region_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_test(
    name = "localization_to_region_calculator_test",
    srcs = ["localization_to_region_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":localization_to_region_calculator",
        ":localization_to_region_calculator_cc_proto",
        "//mediapipe/examples/desktop/autoflip:autoflip_messages_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
)
