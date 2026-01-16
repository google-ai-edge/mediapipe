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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "tone_models_proto",
    srcs = ["tone_models.proto"],
)

mediapipe_proto_library(
    name = "tone_estimation_proto",
    srcs = ["tone_estimation.proto"],
    deps = [":tone_models_proto"],
)

mediapipe_proto_library(
    name = "region_flow_computation_proto",
    srcs = ["region_flow_computation.proto"],
    deps = [
        ":tone_estimation_proto",
    ],
)

mediapipe_proto_library(
    name = "motion_saliency_proto",
    srcs = ["motion_saliency.proto"],
)

mediapipe_proto_library(
    name = "motion_estimation_proto",
    srcs = ["motion_estimation.proto"],
)

mediapipe_proto_library(
    name = "motion_analysis_proto",
    srcs = ["motion_analysis.proto"],
    deps = [
        ":motion_estimation_proto",
        ":motion_saliency_proto",
        ":region_flow_computation_proto",
    ],
)

mediapipe_proto_library(
    name = "region_flow_proto",
    srcs = ["region_flow.proto"],
)

mediapipe_proto_library(
    name = "motion_models_proto",
    srcs = ["motion_models.proto"],
)

mediapipe_proto_library(
    name = "camera_motion_proto",
    srcs = ["camera_motion.proto"],
    deps = [":motion_models_proto"],
)

mediapipe_proto_library(
    name = "push_pull_filtering_proto",
    srcs = ["push_pull_filtering.proto"],
)

mediapipe_proto_library(
    name = "frame_selection_solution_evaluator_proto",
    srcs = ["frame_selection_solution_evaluator.proto"],
)

mediapipe_proto_library(
    name = "frame_selection_proto",
    srcs = ["frame_selection.proto"],
    deps = [
        ":camera_motion_proto",
        ":frame_selection_solution_evaluator_proto",
        ":region_flow_proto",
    ],
)

mediapipe_proto_library(
    name = "flow_packager_proto",
    srcs = ["flow_packager.proto"],
    deps = [
        ":motion_models_proto",
        ":region_flow_proto",
    ],
)

mediapipe_proto_library(
    name = "tracking_proto",
    srcs = ["tracking.proto"],
    deps = [
        ":motion_models_proto",
    ],
)

mediapipe_proto_library(
    name = "box_tracker_proto",
    srcs = ["box_tracker.proto"],
    deps = [":tracking_proto"],
)

mediapipe_proto_library(
    name = "tracked_detection_manager_config_proto",
    srcs = ["tracked_detection_manager_config.proto"],
)

mediapipe_proto_library(
    name = "box_detector_proto",
    srcs = ["box_detector.proto"],
    deps = [
        ":box_tracker_proto",
        ":region_flow_proto",
    ],
)

cc_library(
    name = "motion_models",
    srcs = ["motion_models.cc"],
    hdrs = ["motion_models.h"],
    deps = [
        ":camera_motion_cc_proto",
        ":motion_models_cc_proto",
        ":region_flow_cc_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:singleton",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
        "@eigen//:eigen3",
    ],
)

cc_library(
    name = "motion_models_cv",
    srcs = ["motion_models_cv.cc"],
    hdrs = ["motion_models_cv.h"],
    deps = [
        ":motion_models",
        ":motion_models_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_library(
    name = "region_flow",
    srcs = ["region_flow.cc"],
    hdrs = ["region_flow.h"],
    deps = [
        ":measure_time",
        ":motion_models",
        ":parallel_invoker",
        ":region_flow_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "camera_motion",
    srcs = ["camera_motion.cc"],
    hdrs = ["camera_motion.h"],
    deps = [
        ":camera_motion_cc_proto",
        ":motion_models",
        ":region_flow",
        ":region_flow_cc_proto",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
    ],
)

PARALLEL_COPTS = ["-DPARALLEL_INVOKER_ACTIVE"] + select({
    "//mediapipe:apple": [],
    "//mediapipe:android": [],
    "//mediapipe:emscripten": ["-UPARALLEL_INVOKER_ACTIVE"],
    "//conditions:default": [],
})

PARALLEL_LINKOPTS = []

cc_library(
    name = "measure_time",
    srcs = ["measure_time.cc"],
    hdrs = ["measure_time.h"],
    deps = [
        "//mediapipe/framework/port:integral_types",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "parallel_invoker",
    srcs = ["parallel_invoker.cc"],
    hdrs = ["parallel_invoker.h"],
    copts = PARALLEL_COPTS,
    linkopts = PARALLEL_LINKOPTS,
    deps = [
        ":parallel_invoker_forbid_mixed_active",
        "//mediapipe/framework/port:threadpool",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "parallel_invoker_forbid_mixed_active",
    srcs = ["parallel_invoker_forbid_mixed.cc"],
    copts = PARALLEL_COPTS,
    alwayslink = 1,  # Forces all symbols to be included.
)

cc_library(
    name = "image_util",
    srcs = ["image_util.cc"],
    hdrs = ["image_util.h"],
    deps = [
        ":motion_models",
        ":motion_models_cc_proto",
        ":region_flow",
        ":region_flow_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "streaming_buffer",
    srcs = ["streaming_buffer.cc"],
    hdrs = ["streaming_buffer.h"],
    deps = [
        "//mediapipe/framework/tool:type_util",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "motion_estimation",
    srcs = ["motion_estimation.cc"],
    hdrs = ["motion_estimation.h"],
    copts = PARALLEL_COPTS,
    linkopts = PARALLEL_LINKOPTS,
    deps = [
        ":camera_motion",
        ":camera_motion_cc_proto",
        ":measure_time",
        ":motion_estimation_cc_proto",
        ":motion_models",
        ":motion_models_cc_proto",
        ":parallel_invoker",
        ":region_flow",
        ":region_flow_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@eigen//:eigen3",
    ],
)

cc_library(
    name = "motion_saliency",
    srcs = ["motion_saliency.cc"],
    hdrs = ["motion_saliency.h"],
    deps = [
        ":camera_motion",
        ":measure_time",
        ":motion_saliency_cc_proto",
        ":region_flow",
        ":region_flow_cc_proto",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "push_pull_filtering",
    hdrs = ["push_pull_filtering.h"],
    deps = [
        ":image_util",
        ":push_pull_filtering_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_core",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "tone_models",
    srcs = ["tone_models.cc"],
    hdrs = ["tone_models.h"],
    deps = [
        ":tone_models_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "tone_estimation",
    srcs = ["tone_estimation.cc"],
    hdrs = ["tone_estimation.h"],
    deps = [
        ":motion_models_cc_proto",
        ":region_flow",
        ":region_flow_cc_proto",
        ":tone_estimation_cc_proto",
        ":tone_models",
        ":tone_models_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_library(
    name = "region_flow_computation",
    srcs = ["region_flow_computation.cc"],
    hdrs = ["region_flow_computation.h"],
    copts = PARALLEL_COPTS,
    linkopts = PARALLEL_LINKOPTS,
    deps = [
        ":camera_motion_cc_proto",
        ":image_util",
        ":measure_time",
        ":motion_estimation",
        ":motion_estimation_cc_proto",
        ":motion_models",
        ":motion_models_cc_proto",
        ":parallel_invoker",
        ":region_flow",
        ":region_flow_cc_proto",
        ":region_flow_computation_cc_proto",
        ":tone_estimation",
        ":tone_estimation_cc_proto",
        ":tone_models",
        ":tone_models_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_features2d",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@eigen//:eigen3",
    ],
)

cc_library(
    name = "region_flow_visualization",
    srcs = ["region_flow_visualization.cc"],
    hdrs = ["region_flow_visualization.h"],
    deps = [
        ":measure_time",
        ":motion_models",
        ":parallel_invoker",
        ":region_flow",
        ":region_flow_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "motion_analysis",
    srcs = ["motion_analysis.cc"],
    hdrs = ["motion_analysis.h"],
    deps = [
        ":camera_motion",
        ":camera_motion_cc_proto",
        ":image_util",
        ":measure_time",
        ":motion_analysis_cc_proto",
        ":motion_estimation",
        ":motion_estimation_cc_proto",
        ":motion_saliency",
        ":motion_saliency_cc_proto",
        ":push_pull_filtering",
        ":region_flow",
        ":region_flow_cc_proto",
        ":region_flow_computation",
        ":region_flow_computation_cc_proto",
        ":region_flow_visualization",
        ":streaming_buffer",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "flow_packager",
    srcs = ["flow_packager.cc"],
    hdrs = ["flow_packager.h"],
    deps = [
        ":camera_motion",
        ":camera_motion_cc_proto",
        ":flow_packager_cc_proto",
        ":motion_estimation",
        ":motion_estimation_cc_proto",
        ":motion_models",
        ":motion_models_cc_proto",
        ":region_flow",
        ":region_flow_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tracking",
    srcs = ["tracking.cc"],
    hdrs = ["tracking.h"],
    copts = PARALLEL_COPTS,
    linkopts = PARALLEL_LINKOPTS,
    deps = [
        ":flow_packager_cc_proto",
        ":measure_time",
        ":motion_models",
        ":motion_models_cc_proto",
        ":motion_models_cv",
        ":parallel_invoker",
        ":region_flow",
        ":tracking_cc_proto",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:opencv_calib3d",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@eigen//:eigen3",
    ],
    alwayslink = 1,
)

cc_library(
    name = "box_tracker",
    srcs = ["box_tracker.cc"],
    hdrs = ["box_tracker.h"],
    deps = [
        ":box_tracker_cc_proto",
        ":flow_packager_cc_proto",
        ":measure_time",
        ":tracking",
        ":tracking_cc_proto",
        "//mediapipe/framework/port:integral_types",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:threadpool",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "box_detector",
    srcs = ["box_detector.cc"],
    hdrs = ["box_detector.h"],
    deps = [
        ":box_detector_cc_proto",
        ":box_tracker",
        ":box_tracker_cc_proto",
        ":flow_packager_cc_proto",
        ":measure_time",
        ":tracking",
        "//mediapipe/framework/port:opencv_calib3d",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_features2d",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tracking_visualization_utilities",
    srcs = ["tracking_visualization_utilities.cc"],
    hdrs = ["tracking_visualization_utilities.h"],
    deps = [
        ":box_tracker",
        ":box_tracker_cc_proto",
        ":flow_packager_cc_proto",
        ":tracking",
        ":tracking_cc_proto",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgproc",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "parallel_invoker_test",
    srcs = ["parallel_invoker_test.cc"],
    copts = PARALLEL_COPTS,
    linkopts = PARALLEL_LINKOPTS,
    deps = [
        ":parallel_invoker",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "image_util_test",
    srcs = [
        "image_util_test.cc",
    ],
    deps = [
        ":image_util",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_test(
    name = "motion_models_test",
    srcs = ["motion_models_test.cc"],
    deps = [
        ":motion_estimation",
        ":motion_models",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:vector",
    ],
)

cc_test(
    name = "region_flow_computation_test",
    srcs = ["region_flow_computation_test.cc"],
    copts = PARALLEL_COPTS,
    data = ["testdata/stabilize_test.png"],
    linkopts = PARALLEL_LINKOPTS,
    linkstatic = 1,
    deps = [
        ":region_flow",
        ":region_flow_cc_proto",
        ":region_flow_computation",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "box_tracker_test",
    timeout = "short",
    srcs = ["box_tracker_test.cc"],
    data = glob(["testdata/box_tracker/*"]),
    deps = [
        ":box_tracker",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest_main",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_library(
    name = "tracked_detection",
    srcs = [
        "tracked_detection.cc",
    ],
    hdrs = [
        "tracked_detection.h",
    ],
    deps = [
        "//mediapipe/framework/formats:rect_cc_proto",
        "//mediapipe/framework/port:vector",
        "@com_google_absl//absl/container:node_hash_map",
    ],
)

cc_test(
    name = "tracked_detection_test",
    srcs = [
        "tracked_detection_test.cc",
    ],
    deps = [
        ":tracked_detection",
        "//mediapipe/framework/port:gtest_main",
    ],
)

cc_library(
    name = "tracked_detection_manager",
    srcs = [
        "tracked_detection_manager.cc",
    ],
    hdrs = [
        "tracked_detection_manager.h",
    ],
    deps = [
        ":tracked_detection",
        ":tracked_detection_manager_config_cc_proto",
        "//mediapipe/framework/formats:rect_cc_proto",
        "@com_google_absl//absl/container:node_hash_map",
    ],
)
