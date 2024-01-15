# This contains the MediaPipe Object Detector Task.
#
# This task takes video frames and outputs synchronized frames along with
# the detection results for one or more object categories, using Object Detector.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "object_detector",
    srcs = ["object_detector.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":object_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/processors:detection_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "object_detector_types",
    srcs = [
        "object_detector_options.d.ts",
        "object_detector_result.d.ts",
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
    name = "object_detector_test_lib",
    testonly = True,
    srcs = [
        "object_detector_test.ts",
    ],
    deps = [
        ":object_detector",
        ":object_detector_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/framework/formats:location_data_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
    ],
)

jasmine_node_test(
    name = "object_detector_test",
    tags = ["nomsan"],
    deps = [":object_detector_test_lib"],
)
