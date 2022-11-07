# This contains the MediaPipe Text Classifier Task.
#
# This task takes text input performs Natural Language classification (including
# BERT-based text classification).

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "text_classifier",
    srcs = [
        "text_classifier.ts",
        "text_classifier_options.d.ts",
        "text_classifier_result.d.ts",
    ],
    deps = [
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
        "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:classifications",
        "//mediapipe/tasks/web/components/processors:base_options",
        "//mediapipe/tasks/web/components/processors:classifier_options",
        "//mediapipe/tasks/web/components/processors:classifier_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:wasm_mediapipe_lib_ts",
    ],
)
