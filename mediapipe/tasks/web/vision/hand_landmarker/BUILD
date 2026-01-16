# This contains the MediaPipe Hand Landmarker Task.
#
# This task takes video frames and outputs synchronized frames along with
# the detection results for one or more hand categories, using Hand Landmarker.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "hand_landmarker",
    srcs = ["hand_landmarker.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":hand_landmarker_types",
        ":hand_landmarks_connections",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/framework/formats:landmark_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/tasks/web/components/processors:landmark_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "hand_landmarker_types",
    srcs = [
        "hand_landmark.d.ts",
        "hand_landmarker_options.d.ts",
        "hand_landmarker_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

mediapipe_ts_library(
    name = "hand_landmarker_test_lib",
    testonly = True,
    srcs = [
        "hand_landmarker_test.ts",
    ],
    deps = [
        ":hand_landmarker",
        ":hand_landmarker_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/tasks/web/components/processors:landmark_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
    ],
)

jasmine_node_test(
    name = "hand_landmarker_test",
    srcs = [":hand_landmarker_test_lib"],
    tags = ["nomsan"],
)

mediapipe_ts_library(
    name = "hand_landmarks_connections",
    srcs = ["hand_landmarks_connections.ts"],
    deps = ["//mediapipe/tasks/web/vision/core:types"],
)
