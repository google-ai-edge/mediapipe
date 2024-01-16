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
#

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "flow_to_image_calculator_proto",
    srcs = ["flow_to_image_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "opencv_video_encoder_calculator_proto",
    srcs = ["opencv_video_encoder_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "motion_analysis_calculator_proto",
    srcs = ["motion_analysis_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util/tracking:motion_analysis_proto",
    ],
)

mediapipe_proto_library(
    name = "flow_packager_calculator_proto",
    srcs = ["flow_packager_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util/tracking:flow_packager_proto",
    ],
)

mediapipe_proto_library(
    name = "box_tracker_calculator_proto",
    srcs = ["box_tracker_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util/tracking:box_tracker_proto",
    ],
)

mediapipe_proto_library(
    name = "tracked_detection_manager_calculator_proto",
    srcs = ["tracked_detection_manager_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util/tracking:tracked_detection_manager_config_proto",
    ],
)

mediapipe_proto_library(
    name = "box_detector_calculator_proto",
    srcs = ["box_detector_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util/tracking:box_detector_proto",
    ],
)

mediapipe_proto_library(
    name = "video_pre_stream_calculator_proto",
    srcs = ["video_pre_stream_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "flow_to_image_calculator",
    srcs = ["flow_to_image_calculator.cc"],
    deps = [
        ":flow_to_image_calculator_cc_proto",
        "//mediapipe/calculators/video/tool:flow_quantizer_model",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats/motion:optical_flow_field",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_video_decoder_calculator",
    srcs = ["opencv_video_decoder_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_video_encoder_calculator",
    srcs = ["opencv_video_encoder_calculator.cc"],
    deps = [
        ":opencv_video_encoder_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tvl1_optical_flow_calculator",
    srcs = ["tvl1_optical_flow_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats/motion:optical_flow_field",
        "//mediapipe/framework/port:opencv_video",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_library(
    name = "motion_analysis_calculator",
    srcs = ["motion_analysis_calculator.cc"],
    deps = [
        ":motion_analysis_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tracking:camera_motion",
        "//mediapipe/util/tracking:camera_motion_cc_proto",
        "//mediapipe/util/tracking:frame_selection_cc_proto",
        "//mediapipe/util/tracking:motion_analysis",
        "//mediapipe/util/tracking:motion_estimation",
        "//mediapipe/util/tracking:motion_models",
        "//mediapipe/util/tracking:region_flow_cc_proto",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "flow_packager_calculator",
    srcs = ["flow_packager_calculator.cc"],
    deps = [
        ":flow_packager_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/util/tracking:camera_motion_cc_proto",
        "//mediapipe/util/tracking:flow_packager",
        "//mediapipe/util/tracking:region_flow_cc_proto",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

cc_library(
    name = "box_tracker_calculator",
    srcs = ["box_tracker_calculator.cc"],
    deps = [
        ":box_tracker_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",  # fixdeps: keep -- required for exobazel build.
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:options_util",
        "//mediapipe/util/tracking",
        "//mediapipe/util/tracking:box_tracker",
        "//mediapipe/util/tracking:tracking_visualization_utilities",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "box_detector_calculator",
    srcs = ["box_detector_calculator.cc"],
    deps = [
        ":box_detector_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",  # fixdeps: keep -- required for exobazel build.
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_features2d",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
        "//mediapipe/util/tracking",
        "//mediapipe/util/tracking:box_detector",
        "//mediapipe/util/tracking:box_tracker",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "//mediapipe/util/tracking:flow_packager_cc_proto",
        "//mediapipe/util/tracking:tracking_visualization_utilities",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + select({
        "//mediapipe:android": [
            "//mediapipe/util/android/file/base",
        ],
        "//mediapipe:ios": [
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

cc_library(
    name = "tracked_detection_manager_calculator",
    srcs = ["tracked_detection_manager_calculator.cc"],
    deps = [
        ":tracked_detection_manager_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tracking",
        "//mediapipe/util/tracking:box_tracker",
        "//mediapipe/util/tracking:tracked_detection",
        "//mediapipe/util/tracking:tracked_detection_manager",
        "//mediapipe/util/tracking:tracking_visualization_utilities",
        "@com_google_absl//absl/container:node_hash_map",
    ],
    alwayslink = 1,
)

cc_library(
    name = "video_pre_stream_calculator",
    srcs = ["video_pre_stream_calculator.cc"],
    deps = [
        ":video_pre_stream_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:video_stream_header",
    ],
    alwayslink = 1,
)

filegroup(
    name = "test_videos",
    srcs = [
        "testdata/format_FLV_H264_AAC.video",
        "testdata/format_MKV_VP8_VORBIS.video",
        "testdata/format_MP4_AVC720P_AAC.video",
    ],
)

cc_test(
    name = "opencv_video_decoder_calculator_test",
    srcs = ["opencv_video_decoder_calculator_test.cc"],
    data = [":test_videos"],
    deps = [
        ":opencv_video_decoder_calculator",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:test_util",
    ],
)

cc_test(
    name = "opencv_video_encoder_calculator_test",
    srcs = ["opencv_video_encoder_calculator_test.cc"],
    data = [":test_videos"],
    deps = [
        ":opencv_video_decoder_calculator",
        ":opencv_video_encoder_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:deleting_file",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:test_util",
    ],
)

cc_test(
    name = "tvl1_optical_flow_calculator_test",
    srcs = ["tvl1_optical_flow_calculator_test.cc"],
    linkstatic = 1,
    deps = [
        ":tvl1_optical_flow_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats/motion:optical_flow_field",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

mediapipe_binary_graph(
    name = "parallel_tracker_binarypb",
    graph = "testdata/parallel_tracker_graph.pbtxt",
    output_name = "testdata/parallel_tracker.binarypb",
    deps = [
        ":box_tracker_calculator",
        ":flow_packager_calculator",
        ":motion_analysis_calculator",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
    ],
)

mediapipe_binary_graph(
    name = "tracker_binarypb",
    graph = "testdata/tracker_graph.pbtxt",
    output_name = "testdata/tracker.binarypb",
    deps = [
        ":box_tracker_calculator",
        ":flow_packager_calculator",
        ":motion_analysis_calculator",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
    ],
)

cc_test(
    name = "tracking_graph_test",
    size = "small",
    srcs = ["tracking_graph_test.cc"],
    copts = ["-DPARALLEL_INVOKER_ACTIVE"] + select({
        "//mediapipe:apple": [],
        "//mediapipe:android": [],
        "//conditions:default": [],
    }),
    data = [
        ":testdata/lenna.png",
        ":testdata/parallel_tracker.binarypb",
        ":testdata/tracker.binarypb",
    ],
    deps = [
        ":box_tracker_calculator",
        ":box_tracker_calculator_cc_proto",
        ":flow_packager_calculator",
        ":motion_analysis_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:advanced_proto",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/stream_handler:fixed_size_input_stream_handler",
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        "//mediapipe/framework/tool:test_util",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "//mediapipe/util/tracking:tracking_cc_proto",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_test(
    name = "video_pre_stream_calculator_test",
    srcs = ["video_pre_stream_calculator_test.cc"],
    deps = [
        ":video_pre_stream_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)
