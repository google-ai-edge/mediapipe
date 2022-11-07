# This contains the MediaPipe Audio Classifier Task.
#
# This task takes audio data and outputs the classification result.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "audio_classifier",
    srcs = [
        "audio_classifier.ts",
        "audio_classifier_options.ts",
        "audio_classifier_result.ts",
    ],
    deps = [
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/tasks/cc/audio/audio_classifier/proto:audio_classifier_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
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
