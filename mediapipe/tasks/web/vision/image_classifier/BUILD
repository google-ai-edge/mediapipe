# This contains the MediaPipe Image Classifier Task.
#
# This task takes video or image frames and outputs the classification result.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "image_classifier",
    srcs = ["image_classifier.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":image_classifier_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/image_classifier/proto:image_classifier_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:classification_result",
        "//mediapipe/tasks/web/components/processors:classifier_options",
        "//mediapipe/tasks/web/components/processors:classifier_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "image_classifier_types",
    srcs = [
        "image_classifier_options.d.ts",
        "image_classifier_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:classification_result",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

mediapipe_ts_library(
    name = "image_classifier_test_lib",
    testonly = True,
    srcs = [
        "image_classifier_test.ts",
    ],
    deps = [
        ":image_classifier",
        ":image_classifier_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
    ],
)

jasmine_node_test(
    name = "image_classifier_test",
    tags = ["nomsan"],
    deps = [":image_classifier_test_lib"],
)
