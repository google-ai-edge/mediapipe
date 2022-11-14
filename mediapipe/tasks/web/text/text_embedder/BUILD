# This contains the MediaPipe Text Embedder Task.
#
# This task takes text input and performs embedding
#

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "text_embedder",
    srcs = [
        "text_embedder.ts",
        "text_embedder_options.d.ts",
        "text_embedder_result.d.ts",
    ],
    deps = [
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_jspb_proto",
        "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:embedding_result",
        "//mediapipe/tasks/web/components/processors:base_options",
        "//mediapipe/tasks/web/components/processors:embedder_options",
        "//mediapipe/tasks/web/components/processors:embedder_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:embedder_options",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:wasm_mediapipe_lib_ts",
    ],
)
