# This contains the MediaPipe Audio Embedder Task.
#
# This task takes audio input and performs embedding.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "audio_embedder",
    srcs = ["audio_embedder.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":audio_embedder_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/tasks/cc/audio/audio_embedder/proto:audio_embedder_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/audio/core:audio_task_runner",
        "//mediapipe/tasks/web/components/processors:embedder_options",
        "//mediapipe/tasks/web/components/processors:embedder_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:embedder_options",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "audio_embedder_types",
    srcs = [
        "audio_embedder_options.d.ts",
        "audio_embedder_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:embedding_result",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:embedder_options",
    ],
)

mediapipe_ts_library(
    name = "audio_embedder_test_lib",
    testonly = True,
    srcs = [
        "audio_embedder_test.ts",
    ],
    deps = [
        ":audio_embedder",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
    ],
)

jasmine_node_test(
    name = "audio_embedder_test",
    srcs = [":audio_embedder_test_lib"],
)
