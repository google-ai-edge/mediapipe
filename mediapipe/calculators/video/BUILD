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

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:private"])

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

proto_library(
    name = "flow_to_image_calculator_proto",
    srcs = ["flow_to_image_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "opencv_video_encoder_calculator_proto",
    srcs = ["opencv_video_encoder_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "flow_to_image_calculator_cc_proto",
    srcs = ["flow_to_image_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":flow_to_image_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "opencv_video_encoder_calculator_cc_proto",
    srcs = ["opencv_video_encoder_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":opencv_video_encoder_calculator_proto"],
)

cc_library(
    name = "flow_to_image_calculator",
    srcs = ["flow_to_image_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/calculators/video:flow_to_image_calculator_cc_proto",
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
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "opencv_video_encoder_calculator",
    srcs = ["opencv_video_encoder_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":opencv_video_encoder_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tvl1_optical_flow_calculator",
    srcs = ["tvl1_optical_flow_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats/motion:optical_flow_field",
        "//mediapipe/framework/port:opencv_video",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

cc_test(
    name = "opencv_video_decoder_calculator_test",
    srcs = ["opencv_video_decoder_calculator_test.cc"],
    data = ["//mediapipe/calculators/video/testdata:test_videos"],
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
    ],
)

cc_test(
    name = "opencv_video_encoder_calculator_test",
    srcs = ["opencv_video_encoder_calculator_test.cc"],
    data = ["//mediapipe/calculators/video/testdata:test_videos"],
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
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_test(
    name = "tvl1_optical_flow_calculator_test",
    srcs = ["tvl1_optical_flow_calculator_test.cc"],
    data = ["//mediapipe/calculators/image/testdata:test_images"],
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
