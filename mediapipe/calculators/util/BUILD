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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

proto_library(
    name = "annotation_overlay_calculator_proto",
    srcs = ["annotation_overlay_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
    ],
)

proto_library(
    name = "detection_label_id_to_text_calculator_proto",
    srcs = ["detection_label_id_to_text_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "timed_box_list_id_to_label_calculator_proto",
    srcs = ["timed_box_list_id_to_label_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "latency_proto",
    srcs = ["latency.proto"],
)

proto_library(
    name = "non_max_suppression_calculator_proto",
    srcs = ["non_max_suppression_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "packet_frequency_proto",
    srcs = ["packet_frequency.proto"],
)

proto_library(
    name = "packet_frequency_calculator_proto",
    srcs = ["packet_frequency_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "packet_latency_calculator_proto",
    srcs = ["packet_latency_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "collection_has_min_size_calculator_proto",
    srcs = ["collection_has_min_size_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "association_calculator_proto",
    srcs = ["association_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "annotation_overlay_calculator_cc_proto",
    srcs = ["annotation_overlay_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":annotation_overlay_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "detection_label_id_to_text_calculator_cc_proto",
    srcs = ["detection_label_id_to_text_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":detection_label_id_to_text_calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "timed_box_list_id_to_label_calculator_cc_proto",
    srcs = ["timed_box_list_id_to_label_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":timed_box_list_id_to_label_calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "latency_cc_proto",
    srcs = ["latency.proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":latency_proto"],
)

mediapipe_cc_proto_library(
    name = "non_max_suppression_calculator_cc_proto",
    srcs = ["non_max_suppression_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":non_max_suppression_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "packet_frequency_cc_proto",
    srcs = ["packet_frequency.proto"],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":packet_frequency_proto"],
)

mediapipe_cc_proto_library(
    name = "packet_frequency_calculator_cc_proto",
    srcs = ["packet_frequency_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [
        ":packet_frequency_calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "packet_latency_calculator_cc_proto",
    srcs = ["packet_latency_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [
        ":packet_latency_calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "collection_has_min_size_calculator_cc_proto",
    srcs = ["collection_has_min_size_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":collection_has_min_size_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "association_calculator_cc_proto",
    srcs = ["association_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//mediapipe:__subpackages__"],
    deps = [":association_calculator_proto"],
)

cc_library(
    name = "packet_frequency_calculator",
    srcs = ["packet_frequency_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/calculators/util:packet_frequency_calculator_cc_proto",
        "//mediapipe/calculators/util:packet_frequency_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_frequency_calculator_test",
    size = "small",
    srcs = ["packet_frequency_calculator_test.cc"],
    deps = [
        ":packet_frequency_calculator",
        "//mediapipe/calculators/util:packet_frequency_cc_proto",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
    ],
)

cc_library(
    name = "packet_latency_calculator",
    srcs = ["packet_latency_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/calculators/util:latency_cc_proto",
        "//mediapipe/calculators/util:packet_latency_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_test(
    name = "packet_latency_calculator_test",
    size = "small",
    srcs = ["packet_latency_calculator_test.cc"],
    deps = [
        ":packet_latency_calculator",
        "//mediapipe/calculators/util:latency_cc_proto",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/stream_handler:immediate_input_stream_handler",
        "//mediapipe/framework/tool:simulation_clock",
        "//mediapipe/framework/tool:simulation_clock_executor",
        "//mediapipe/framework/tool:sink",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "clock_timestamp_calculator",
    srcs = ["clock_timestamp_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/deps:clock",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library(
    name = "clock_latency_calculator",
    srcs = ["clock_latency_calculator.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library(
    name = "annotation_overlay_calculator",
    srcs = ["annotation_overlay_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_overlay_calculator_cc_proto",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "//mediapipe/util:annotation_renderer",
        "//mediapipe/util:render_data_cc_proto",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
            "//mediapipe/gpu:gl_simple_shaders",
            "//mediapipe/gpu:gpu_buffer",
            "//mediapipe/gpu:shader_util",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "detection_label_id_to_text_calculator",
    srcs = ["detection_label_id_to_text_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":detection_label_id_to_text_calculator_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/util:resource_util",
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
    name = "timed_box_list_id_to_label_calculator",
    srcs = ["timed_box_list_id_to_label_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":timed_box_list_id_to_label_calculator_cc_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "//mediapipe/util:resource_util",
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
    name = "non_max_suppression_calculator",
    srcs = ["non_max_suppression_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":non_max_suppression_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

mediapipe_cc_proto_library(
    name = "thresholding_calculator_cc_proto",
    srcs = ["thresholding_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":thresholding_calculator_proto"],
)

cc_library(
    name = "thresholding_calculator",
    srcs = ["thresholding_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":thresholding_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

mediapipe_cc_proto_library(
    name = "landmarks_to_detection_calculator_cc_proto",
    srcs = ["landmarks_to_detection_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":landmarks_to_detection_calculator_proto"],
)

cc_library(
    name = "landmarks_to_detection_calculator",
    srcs = ["landmarks_to_detection_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":landmarks_to_detection_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

mediapipe_cc_proto_library(
    name = "detections_to_rects_calculator_cc_proto",
    srcs = ["detections_to_rects_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":detections_to_rects_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "landmark_projection_calculator_cc_proto",
    srcs = ["landmark_projection_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":landmark_projection_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "landmarks_to_floats_calculator_cc_proto",
    srcs = ["landmarks_to_floats_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":landmarks_to_floats_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "rect_transformation_calculator_cc_proto",
    srcs = ["rect_transformation_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":rect_transformation_calculator_proto"],
)

cc_library(
    name = "detections_to_rects_calculator",
    srcs = [
        "detections_to_rects_calculator.cc",
    ],
    hdrs = [
        "detections_to_rects_calculator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":detections_to_rects_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rect_transformation_calculator",
    srcs = ["rect_transformation_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":rect_transformation_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detections_to_rects_calculator_test",
    size = "small",
    srcs = ["detections_to_rects_calculator_test.cc"],
    deps = [
        ":detections_to_rects_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

proto_library(
    name = "rect_to_render_data_calculator_proto",
    srcs = ["rect_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

proto_library(
    name = "detections_to_render_data_calculator_proto",
    srcs = ["detections_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

proto_library(
    name = "landmarks_to_render_data_calculator_proto",
    srcs = ["landmarks_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

proto_library(
    name = "timed_box_list_to_render_data_calculator_proto",
    srcs = ["timed_box_list_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

proto_library(
    name = "labels_to_render_data_calculator_proto",
    srcs = ["labels_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

proto_library(
    name = "thresholding_calculator_proto",
    srcs = ["thresholding_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

proto_library(
    name = "detections_to_rects_calculator_proto",
    srcs = ["detections_to_rects_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "landmark_projection_calculator_proto",
    srcs = ["landmark_projection_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "landmarks_to_floats_calculator_proto",
    srcs = ["landmarks_to_floats_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "rect_transformation_calculator_proto",
    srcs = ["rect_transformation_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "landmarks_to_detection_calculator_proto",
    srcs = ["landmarks_to_detection_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "rect_to_render_data_calculator_cc_proto",
    srcs = ["rect_to_render_data_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":rect_to_render_data_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "detections_to_render_data_calculator_cc_proto",
    srcs = ["detections_to_render_data_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":detections_to_render_data_calculator_proto"],
)

cc_library(
    name = "detections_to_render_data_calculator",
    srcs = ["detections_to_render_data_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":detections_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

mediapipe_cc_proto_library(
    name = "landmarks_to_render_data_calculator_cc_proto",
    srcs = ["landmarks_to_render_data_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":landmarks_to_render_data_calculator_proto"],
)

cc_library(
    name = "landmarks_to_render_data_calculator",
    srcs = ["landmarks_to_render_data_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":landmarks_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

mediapipe_cc_proto_library(
    name = "timed_box_list_to_render_data_calculator_cc_proto",
    srcs = ["timed_box_list_to_render_data_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":timed_box_list_to_render_data_calculator_proto"],
)

cc_library(
    name = "timed_box_list_to_render_data_calculator",
    srcs = ["timed_box_list_to_render_data_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":timed_box_list_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "//mediapipe/util/tracking:tracking_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "labels_to_render_data_calculator",
    srcs = ["labels_to_render_data_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":labels_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rect_to_render_data_calculator",
    srcs = ["rect_to_render_data_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":rect_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detections_to_render_data_calculator_test",
    size = "small",
    srcs = ["detections_to_render_data_calculator_test.cc"],
    deps = [
        ":detections_to_render_data_calculator",
        ":detections_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "detection_letterbox_removal_calculator",
    srcs = ["detection_letterbox_removal_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "landmark_letterbox_removal_calculator",
    srcs = ["landmark_letterbox_removal_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "landmark_projection_calculator",
    srcs = ["landmark_projection_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":landmark_projection_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "landmarks_to_floats_calculator",
    srcs = ["landmarks_to_floats_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":landmarks_to_floats_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@eigen_archive//:eigen",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detection_letterbox_removal_calculator_test",
    srcs = ["detection_letterbox_removal_calculator_test.cc"],
    deps = [
        ":detection_letterbox_removal_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
    ],
)

cc_test(
    name = "landmark_letterbox_removal_calculator_test",
    srcs = ["landmark_letterbox_removal_calculator_test.cc"],
    deps = [
        ":landmark_letterbox_removal_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/tool:validate_type",
    ],
)

proto_library(
    name = "top_k_scores_calculator_proto",
    srcs = ["top_k_scores_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "top_k_scores_calculator_cc_proto",
    srcs = ["top_k_scores_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":top_k_scores_calculator_proto"],
)

cc_library(
    name = "top_k_scores_calculator",
    srcs = ["top_k_scores_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":top_k_scores_calculator_cc_proto",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/util:resource_util",
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

cc_test(
    name = "top_k_scores_calculator_test",
    srcs = ["top_k_scores_calculator_test.cc"],
    deps = [
        ":top_k_scores_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

mediapipe_cc_proto_library(
    name = "labels_to_render_data_calculator_cc_proto",
    srcs = ["labels_to_render_data_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":labels_to_render_data_calculator_proto"],
)

cc_library(
    name = "local_file_contents_calculator",
    srcs = ["local_file_contents_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:resource_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "local_file_pattern_contents_calculator",
    srcs = ["local_file_pattern_contents_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "filter_collection_calculator",
    srcs = ["filter_collection_calculator.cc"],
    hdrs = ["filter_collection_calculator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "collection_has_min_size_calculator",
    srcs = ["collection_has_min_size_calculator.cc"],
    hdrs = ["collection_has_min_size_calculator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":collection_has_min_size_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "collection_has_min_size_calculator_test",
    srcs = ["collection_has_min_size_calculator_test.cc"],
    deps = [
        ":collection_has_min_size_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_library(
    name = "association_calculator",
    hdrs = ["association_calculator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":association_calculator_cc_proto",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "association_norm_rect_calculator",
    srcs = ["association_norm_rect_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":association_calculator",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "association_detection_calculator",
    srcs = ["association_detection_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":association_calculator",
        "//mediapipe/framework:calculator_context",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/port:rectangle",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "association_calculator_test",
    srcs = ["association_calculator_test.cc"],
    deps = [
        ":association_detection_calculator",
        ":association_norm_rect_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:collection_item_id",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/deps:message_matchers",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

cc_library(
    name = "detections_to_timed_box_list_calculator",
    srcs = ["detections_to_timed_box_list_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tracking:box_tracker",
    ],
    alwayslink = 1,
)

cc_library(
    name = "detection_unique_id_calculator",
    srcs = ["detection_unique_id_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)
