# Copyright 2020 The MediaPipe Authors.
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

load("//mediapipe/framework:mediapipe_register_type.bzl", "mediapipe_register_type")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "object_proto",
    srcs = ["object.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "a_r_capture_metadata_proto",
    srcs = ["a_r_capture_metadata.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "annotation_proto",
    srcs = ["annotation_data.proto"],
    def_options_lib = False,
    visibility = ["//visibility:public"],
    deps = [
        ":a_r_capture_metadata_proto",
        ":object_proto",
    ],
)

mediapipe_register_type(
    base_name = "annotation",
    include_headers = ["mediapipe/modules/objectron/calculators/annotation_data.pb.h"],
    types = [
        "::mediapipe::FrameAnnotation",
    ],
    deps = [":annotation_cc_proto"],
)

mediapipe_proto_library(
    name = "camera_parameters_proto",
    srcs = ["camera_parameters.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "frame_annotation_tracker_calculator_proto",
    srcs = ["frame_annotation_tracker_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "belief_decoder_config_proto",
    srcs = ["belief_decoder_config.proto"],
    visibility = ["//visibility:public"],
)

mediapipe_proto_library(
    name = "tflite_tensors_to_objects_calculator_proto",
    srcs = ["tflite_tensors_to_objects_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":belief_decoder_config_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "tensors_to_objects_calculator_proto",
    srcs = ["tensors_to_objects_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":belief_decoder_config_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "lift_2d_frame_annotation_to_3d_calculator_proto",
    srcs = ["lift_2d_frame_annotation_to_3d_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":belief_decoder_config_proto",
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "frame_annotation_to_rect_calculator_proto",
    srcs = ["frame_annotation_to_rect_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "filter_detection_calculator_proto",
    srcs = ["filter_detection_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "box_util",
    srcs = ["box_util.cc"],
    hdrs = ["box_util.h"],
    deps = [
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "frame_annotation_tracker",
    srcs = ["frame_annotation_tracker.cc"],
    hdrs = ["frame_annotation_tracker.h"],
    deps = [
        ":annotation_cc_proto",
        ":box_util",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "epnp",
    srcs = [
        "epnp.cc",
    ],
    hdrs = [
        "epnp.h",
    ],
    deps = [
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "decoder",
    srcs = [
        "decoder.cc",
    ],
    hdrs = [
        "decoder.h",
    ],
    deps = [
        ":annotation_cc_proto",
        ":belief_decoder_config_cc_proto",
        ":box",
        ":epnp",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "tensor_util",
    srcs = [
        "tensor_util.cc",
    ],
    hdrs = [
        "tensor_util.h",
    ],
    deps = [
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "@com_google_absl//absl/log:absl_check",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "box",
    srcs = [
        "box.cc",
        "model.cc",
    ],
    hdrs = [
        "box.h",
        "model.h",
        "types.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":object_cc_proto",
        "//mediapipe/framework/port:logging",
        "@com_google_absl//absl/log:absl_check",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "frame_annotation_to_timed_box_list_calculator",
    srcs = ["frame_annotation_to_timed_box_list_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":box_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "frame_annotation_tracker_calculator",
    srcs = ["frame_annotation_tracker_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":frame_annotation_tracker",
        ":frame_annotation_tracker_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tflite_tensors_to_objects_calculator",
    srcs = ["tflite_tensors_to_objects_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":belief_decoder_config_cc_proto",
        ":decoder",
        ":tensor_util",
        ":tflite_tensors_to_objects_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensors_to_objects_calculator",
    srcs = ["tensors_to_objects_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":belief_decoder_config_cc_proto",
        ":decoder",
        ":tensor_util",
        ":tensors_to_objects_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:tensor",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lift_2d_frame_annotation_to_3d_calculator",
    srcs = ["lift_2d_frame_annotation_to_3d_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":decoder",
        ":lift_2d_frame_annotation_to_3d_calculator_cc_proto",
        ":tensor_util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "frame_annotation_to_rect_calculator",
    srcs = ["frame_annotation_to_rect_calculator.cc"],
    deps = [
        ":annotation_cc_proto",
        ":frame_annotation_to_rect_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "landmarks_to_frame_annotation_calculator",
    srcs = ["landmarks_to_frame_annotation_calculator.cc"],
    deps = [
        ":annotation_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "filter_detection_calculator",
    srcs = ["filter_detection_calculator.cc"],
    deps = [
        ":filter_detection_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:re2",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_test(
    name = "filter_detection_calculator_test",
    srcs = ["filter_detection_calculator_test.cc"],
    deps = [
        ":filter_detection_calculator",  # build_cleaner: keep
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework:calculator_runner",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/util:packet_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "box_util_test",
    srcs = ["box_util_test.cc"],
    deps = [
        ":box_util",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
    ],
)

cc_test(
    name = "frame_annotation_tracker_test",
    srcs = ["frame_annotation_tracker_test.cc"],
    deps = [
        ":annotation_cc_proto",
        ":frame_annotation_tracker",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:logging",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
    ],
)
