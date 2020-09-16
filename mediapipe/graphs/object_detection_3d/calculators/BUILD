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

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "object_proto",
    srcs = [
        "object.proto",
    ],
)

proto_library(
    name = "a_r_capture_metadata_proto",
    srcs = [
        "a_r_capture_metadata.proto",
    ],
)

proto_library(
    name = "annotation_proto",
    srcs = [
        "annotation_data.proto",
    ],
    deps = [
        ":a_r_capture_metadata_proto",
        ":object_proto",
    ],
)

proto_library(
    name = "belief_decoder_config_proto",
    srcs = [
        "belief_decoder_config.proto",
    ],
)

proto_library(
    name = "camera_parameters_proto",
    srcs = [
        "camera_parameters.proto",
    ],
)

proto_library(
    name = "frame_annotation_tracker_calculator_proto",
    srcs = ["frame_annotation_tracker_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "gl_animation_overlay_calculator_proto",
    srcs = ["gl_animation_overlay_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_proto"],
)

proto_library(
    name = "tflite_tensors_to_objects_calculator_proto",
    srcs = ["tflite_tensors_to_objects_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":belief_decoder_config_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "lift_2d_frame_annotation_to_3d_calculator_proto",
    srcs = ["lift_2d_frame_annotation_to_3d_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        ":belief_decoder_config_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "annotations_to_model_matrices_calculator_proto",
    srcs = ["annotations_to_model_matrices_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "model_matrix_proto",
    srcs = ["model_matrix.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
    ],
)

proto_library(
    name = "annotations_to_render_data_calculator_proto",
    srcs = ["annotations_to_render_data_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_proto",
        "//mediapipe/util:color_proto",
    ],
)

mediapipe_cc_proto_library(
    name = "object_cc_proto",
    srcs = ["object.proto"],
    visibility = ["//visibility:public"],
    deps = [":object_proto"],
)

mediapipe_cc_proto_library(
    name = "a_r_capture_metadata_cc_proto",
    srcs = ["a_r_capture_metadata.proto"],
    visibility = ["//visibility:public"],
    deps = [":a_r_capture_metadata_proto"],
)

mediapipe_cc_proto_library(
    name = "annotation_cc_proto",
    srcs = ["annotation_data.proto"],
    cc_deps = [
        ":a_r_capture_metadata_cc_proto",
        ":object_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":annotation_proto"],
)

mediapipe_cc_proto_library(
    name = "camera_parameters_cc_proto",
    srcs = ["camera_parameters.proto"],
    visibility = ["//visibility:public"],
    deps = [":camera_parameters_proto"],
)

mediapipe_cc_proto_library(
    name = "frame_annotation_tracker_calculator_cc_proto",
    srcs = ["frame_annotation_tracker_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":frame_annotation_tracker_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "gl_animation_overlay_calculator_cc_proto",
    srcs = ["gl_animation_overlay_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":gl_animation_overlay_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "belief_decoder_config_cc_proto",
    srcs = ["belief_decoder_config.proto"],
    visibility = ["//visibility:public"],
    deps = [":belief_decoder_config_proto"],
)

mediapipe_cc_proto_library(
    name = "tflite_tensors_to_objects_calculator_cc_proto",
    srcs = ["tflite_tensors_to_objects_calculator.proto"],
    cc_deps = [
        ":belief_decoder_config_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":tflite_tensors_to_objects_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "lift_2d_frame_annotation_to_3d_calculator_cc_proto",
    srcs = ["lift_2d_frame_annotation_to_3d_calculator.proto"],
    cc_deps = [
        ":belief_decoder_config_cc_proto",
        "//mediapipe/framework:calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":lift_2d_frame_annotation_to_3d_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "annotations_to_model_matrices_calculator_cc_proto",
    srcs = ["annotations_to_model_matrices_calculator.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":annotations_to_model_matrices_calculator_proto"],
)

mediapipe_cc_proto_library(
    name = "model_matrix_cc_proto",
    srcs = ["model_matrix.proto"],
    cc_deps = ["//mediapipe/framework:calculator_cc_proto"],
    visibility = ["//visibility:public"],
    deps = [":model_matrix_proto"],
)

mediapipe_cc_proto_library(
    name = "annotations_to_render_data_calculator_cc_proto",
    srcs = ["annotations_to_render_data_calculator.proto"],
    cc_deps = [
        "//mediapipe/framework:calculator_cc_proto",
        "//mediapipe/util:color_cc_proto",
    ],
    visibility = ["//visibility:public"],
    deps = [":annotations_to_render_data_calculator_proto"],
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
        "//mediapipe/framework/port:logging",
        "//mediapipe/util/tracking:box_tracker_cc_proto",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "gl_animation_overlay_calculator",
    srcs = ["gl_animation_overlay_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":camera_parameters_cc_proto",
        ":gl_animation_overlay_calculator_cc_proto",
        ":model_matrix_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:shader_util",
        "//mediapipe/util/android:asset_manager_util",
    ],
    alwayslink = 1,
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
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen",
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
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
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
        "@eigen_archive//:eigen",
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
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lift_2d_frame_annotation_to_3d_calculator",
    srcs = ["lift_2d_frame_annotation_to_3d_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":belief_decoder_config_cc_proto",
        ":decoder",
        ":lift_2d_frame_annotation_to_3d_calculator_cc_proto",
        ":tensor_util",
        ":tflite_tensors_to_objects_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:ret_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "annotations_to_model_matrices_calculator",
    srcs = ["annotations_to_model_matrices_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":annotations_to_model_matrices_calculator_cc_proto",
        ":box",
        ":model_matrix_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/util:color_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen",
    ],
    alwayslink = 1,
)

cc_library(
    name = "annotations_to_render_data_calculator",
    srcs = ["annotations_to_render_data_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":annotation_cc_proto",
        ":annotations_to_render_data_calculator_cc_proto",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
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
    ],
)
