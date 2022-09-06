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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "alignment_points_to_rects_calculator",
    srcs = ["alignment_points_to_rects_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/util:detections_to_rects_calculator",
        "//mediapipe/calculators/util:detections_to_rects_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "annotation_overlay_calculator_proto",
    srcs = ["annotation_overlay_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
    ],
)

mediapipe_proto_library(
    name = "detection_label_id_to_text_calculator_proto",
    srcs = ["detection_label_id_to_text_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:label_map_proto",
    ],
)

mediapipe_proto_library(
    name = "filter_detections_calculator_proto",
    srcs = ["filter_detections_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "timed_box_list_id_to_label_calculator_proto",
    srcs = ["timed_box_list_id_to_label_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "latency_proto",
    srcs = ["latency.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "non_max_suppression_calculator_proto",
    srcs = ["non_max_suppression_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "packet_frequency_proto",
    srcs = ["packet_frequency.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "packet_frequency_calculator_proto",
    srcs = ["packet_frequency_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "packet_latency_calculator_proto",
    srcs = ["packet_latency_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "collection_has_min_size_calculator_proto",
    srcs = ["collection_has_min_size_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "association_calculator_proto",
    srcs = ["association_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
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

cc_test(
    name = "filter_detections_calculator_test",
    size = "small",
    srcs = ["filter_detections_calculator_test.cc"],
    deps = [
        ":filter_detections_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
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
        "@com_google_absl//absl/strings",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
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
        "@com_google_absl//absl/container:node_hash_map",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:packet",
        "//mediapipe/util:resource_util",
        "//mediapipe/util:label_map_cc_proto",
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
        "@com_google_absl//absl/container:node_hash_map",
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
    name = "detection_transformation_calculator",
    srcs = ["detection_transformation_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:packet",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detection_transformation_calculator_test",
    size = "small",
    srcs = ["detection_transformation_calculator_test.cc"],
    deps = [
        ":detection_transformation_calculator",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
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

cc_library(
    name = "detection_to_landmarks_calculator",
    srcs = ["detection_to_landmarks_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "filter_detections_calculator",
    srcs = ["filter_detections_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":filter_detections_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
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

cc_library(
    name = "rect_projection_calculator",
    srcs = ["rect_projection_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
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
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

mediapipe_proto_library(
    name = "rect_to_render_data_calculator_proto",
    srcs = ["rect_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

mediapipe_proto_library(
    name = "rect_to_render_scale_calculator_proto",
    srcs = ["rect_to_render_scale_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "detections_to_render_data_calculator_proto",
    srcs = ["detections_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

mediapipe_proto_library(
    name = "landmarks_to_render_data_calculator_proto",
    srcs = ["landmarks_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

mediapipe_proto_library(
    name = "timed_box_list_to_render_data_calculator_proto",
    srcs = ["timed_box_list_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

mediapipe_proto_library(
    name = "labels_to_render_data_calculator_proto",
    srcs = ["labels_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

mediapipe_proto_library(
    name = "thresholding_calculator_proto",
    srcs = ["thresholding_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
)

mediapipe_proto_library(
    name = "detections_to_rects_calculator_proto",
    srcs = ["detections_to_rects_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "landmark_projection_calculator_proto",
    srcs = ["landmark_projection_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "landmark_visibility_calculator",
    srcs = ["landmark_visibility_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

cc_library(
    name = "set_landmark_visibility_calculator",
    srcs = ["set_landmark_visibility_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "landmarks_to_floats_calculator_proto",
    srcs = ["landmarks_to_floats_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "rect_transformation_calculator_proto",
    srcs = ["rect_transformation_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "landmarks_to_detection_calculator_proto",
    srcs = ["landmarks_to_detection_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
        "//mediapipe/util:render_data_proto",
    ],
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

cc_library(
    name = "landmarks_to_render_data_calculator",
    srcs = ["landmarks_to_render_data_calculator.cc"],
    hdrs = ["landmarks_to_render_data_calculator.h"],
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

cc_library(
    name = "rect_to_render_scale_calculator",
    srcs = ["rect_to_render_scale_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":rect_to_render_scale_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
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
    name = "detection_projection_calculator",
    srcs = ["detection_projection_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:point",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detection_projection_calculator_test",
    srcs = ["detection_projection_calculator_test.cc"],
    deps = [
        ":detection_projection_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:point",
    ],
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

cc_test(
    name = "landmark_projection_calculator_test",
    srcs = ["landmark_projection_calculator_test.cc"],
    deps = [
        ":landmark_projection_calculator",
        "//mediapipe/calculators/tensor:image_to_tensor_utils",
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "world_landmark_projection_calculator",
    srcs = ["world_landmark_projection_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "landmarks_smoothing_calculator_proto",
    srcs = ["landmarks_smoothing_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "landmarks_smoothing_calculator",
    srcs = ["landmarks_smoothing_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":landmarks_smoothing_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util/filtering:one_euro_filter",
        "//mediapipe/util/filtering:relative_velocity_filter",
        "@com_google_absl//absl/algorithm:container",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "visibility_smoothing_calculator_proto",
    srcs = ["visibility_smoothing_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "visibility_smoothing_calculator",
    srcs = ["visibility_smoothing_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":visibility_smoothing_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util/filtering:low_pass_filter",
        "@com_google_absl//absl/algorithm:container",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "visibility_copy_calculator_proto",
    srcs = ["visibility_copy_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "visibility_copy_calculator",
    srcs = ["visibility_copy_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":visibility_copy_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/algorithm:container",
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
        "@eigen_archive//:eigen3",
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

mediapipe_proto_library(
    name = "top_k_scores_calculator_proto",
    srcs = ["top_k_scores_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "top_k_scores_calculator",
    srcs = ["top_k_scores_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":top_k_scores_calculator_cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
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
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ],
)

mediapipe_proto_library(
    name = "local_file_contents_calculator_proto",
    srcs = ["local_file_contents_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "local_file_contents_calculator",
    srcs = ["local_file_contents_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":local_file_contents_calculator_cc_proto",
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
        "//mediapipe/framework/port:integral_types",
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
        "//mediapipe/framework/formats:classification_cc_proto",
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
        "//mediapipe/util:rectangle_util",
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

mediapipe_proto_library(
    name = "logic_calculator_proto",
    srcs = ["logic_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "logic_calculator",
    srcs = ["logic_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":logic_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
    ],
    alwayslink = 1,
)

cc_library(
    name = "to_image_calculator",
    srcs = ["to_image_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "from_image_calculator",
    srcs = ["from_image_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
    ] + select({
        "//mediapipe/gpu:disable_gpu": [],
        "//conditions:default": [
            "//mediapipe/gpu:gl_calculator_helper",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "detection_classifications_merger_calculator",
    srcs = ["detection_classifications_merger_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "detection_classifications_merger_calculator_test",
    srcs = ["detection_classifications_merger_calculator_test.cc"],
    deps = [
        ":detection_classifications_merger_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
    ],
)

mediapipe_proto_library(
    name = "refine_landmarks_from_heatmap_calculator_proto",
    srcs = ["refine_landmarks_from_heatmap_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "refine_landmarks_from_heatmap_calculator",
    srcs = ["refine_landmarks_from_heatmap_calculator.cc"],
    hdrs = ["refine_landmarks_from_heatmap_calculator.h"],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":refine_landmarks_from_heatmap_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:statusor",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "landmarks_refinement_calculator_proto",
    srcs = ["landmarks_refinement_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "landmarks_refinement_calculator",
    srcs = ["landmarks_refinement_calculator.cc"],
    hdrs = ["landmarks_refinement_calculator.h"],
    deps = [
        ":landmarks_refinement_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:core_proto",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_test(
    name = "refine_landmarks_from_heatmap_calculator_test",
    srcs = ["refine_landmarks_from_heatmap_calculator_test.cc"],
    deps = [
        ":refine_landmarks_from_heatmap_calculator",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "inverse_matrix_calculator",
    srcs = ["inverse_matrix_calculator.cc"],
    hdrs = ["inverse_matrix_calculator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/api2:node",
        "//mediapipe/framework/api2:port",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = True,
)

cc_test(
    name = "inverse_matrix_calculator_test",
    srcs = ["inverse_matrix_calculator_test.cc"],
    tags = ["desktop_only_test"],
    deps = [
        ":inverse_matrix_calculator",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:parse_text_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)
