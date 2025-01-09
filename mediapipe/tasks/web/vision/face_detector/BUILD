# This contains the MediaPipe Face Detector Task.
#
# This task takes video frames and outputs synchronized frames along with
# the detection results for one or more faces, using Face Detector.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "face_detector",
    srcs = ["face_detector.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":face_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/processors:detection_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "face_detector_types",
    srcs = [
        "face_detector_options.d.ts",
        "face_detector_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:bounding_box",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:detection_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

mediapipe_ts_library(
    name = "face_detector_test_lib",
    testonly = True,
    srcs = [
        "face_detector_test.ts",
    ],
    deps = [
        ":face_detector",
        ":face_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/framework/formats:location_data_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
    ],
)

jasmine_node_test(
    name = "face_detector_test",
    tags = ["nomsan"],
    deps = [":face_detector_test_lib"],
)
