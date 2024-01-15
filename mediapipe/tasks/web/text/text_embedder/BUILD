# This contains the MediaPipe Text Embedder Task.
#
# This task takes text input and performs embedding
#

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "text_embedder",
    srcs = ["text_embedder.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":text_embedder_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:embedding_result",
        "//mediapipe/tasks/web/components/processors:embedder_options",
        "//mediapipe/tasks/web/components/processors:embedder_result",
        "//mediapipe/tasks/web/components/utils:cosine_similarity",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:embedder_options",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "text_embedder_types",
    srcs = [
        "text_embedder_options.d.ts",
        "text_embedder_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:embedding_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:embedder_options",
    ],
)

mediapipe_ts_library(
    name = "text_embedder_test_lib",
    testonly = True,
    srcs = [
        "text_embedder_test.ts",
    ],
    deps = [
        ":text_embedder",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
    ],
)

jasmine_node_test(
    name = "text_embedder_test",
    deps = [":text_embedder_test_lib"],
)
