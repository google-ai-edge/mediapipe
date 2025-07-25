# This contains the MediaPipe Face Landmarker Task.
#
# This task takes video frames and outputs synchronized frames along with
# the detection results for one or more face categories, using Face Landmarker.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "face_landmarker",
    srcs = ["face_landmarker.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":face_landmarker_types",
        ":face_landmarks_connections",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/framework/formats:landmark_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_jspb_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:classification_result",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/tasks/web/components/containers:matrix",
        "//mediapipe/tasks/web/components/processors:classifier_result",
        "//mediapipe/tasks/web/components/processors:landmark_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_library(
    name = "face_landmarks_connections",
    srcs = ["face_landmarks_connections.ts"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/tasks/web/vision/core:types"],
)

mediapipe_ts_declaration(
    name = "face_landmarker_types",
    srcs = [
        "face_landmarker_options.d.ts",
        "face_landmarker_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:classification_result",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/tasks/web/components/containers:matrix",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

mediapipe_ts_library(
    name = "face_landmarker_test_lib",
    testonly = True,
    srcs = [
        "face_landmarker_test.ts",
    ],
    deps = [
        ":face_landmarker",
        ":face_landmarker_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/framework/formats:matrix_data_jspb_proto",
        "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_jspb_proto",
        "//mediapipe/tasks/web/components/processors:landmark_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
    ],
)

jasmine_node_test(
    name = "face_landmarker_test",
    srcs = [":face_landmarker_test_lib"],
    tags = ["nomsan"],
)
