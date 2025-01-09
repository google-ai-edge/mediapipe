# This contains the MediaPipe Pose Landmarker Task.
#
# This task takes video frames and outputs synchronized frames along with
# the detection results for one or more pose categories, using Pose Landmarker.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "pose_landmarker",
    srcs = [
        "pose_landmarker.ts",
        "pose_landmarker_result.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pose_landmarker_types",
        ":pose_landmarks_connections",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:landmark_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/pose_detector/proto:pose_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarker_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarks_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/tasks/web/components/processors:landmark_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:mask",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "pose_landmarker_types",
    srcs = ["pose_landmarker_options.d.ts"],
    deps = ["//mediapipe/tasks/web/vision/core:vision_task_options"],
)

mediapipe_ts_library(
    name = "pose_landmarker_test_lib",
    testonly = True,
    srcs = [
        "pose_landmarker_test.ts",
    ],
    deps = [
        ":pose_landmarker",
        ":pose_landmarker_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/tasks/web/components/processors:landmark_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:mask",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
    ],
)

jasmine_node_test(
    name = "pose_landmarker_test",
    tags = ["nomsan"],
    deps = [":pose_landmarker_test_lib"],
)

mediapipe_ts_library(
    name = "pose_landmarks_connections",
    srcs = ["pose_landmarks_connections.ts"],
    deps = ["//mediapipe/tasks/web/vision/core:types"],
)
