# This contains the MediaPipe Text Classifier Task.
#
# This task takes text input performs Natural Language classification (including
# BERT-based text classification).

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "text_classifier",
    srcs = ["text_classifier.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":text_classifier_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:classification_result",
        "//mediapipe/tasks/web/components/processors:classifier_options",
        "//mediapipe/tasks/web/components/processors:classifier_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "text_classifier_types",
    srcs = [
        "text_classifier_options.d.ts",
        "text_classifier_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:classification_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
    ],
)

mediapipe_ts_library(
    name = "text_classifier_test_lib",
    testonly = True,
    srcs = [
        "text_classifier_test.ts",
    ],
    deps = [
        ":text_classifier",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
    ],
)

jasmine_node_test(
    name = "text_classifier_test",
    deps = [":text_classifier_test_lib"],
)
