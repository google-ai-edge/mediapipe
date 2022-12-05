# This package contains options shared by all MediaPipe Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_declaration(
    name = "core",
    srcs = [
        "task_runner_options.d.ts",
        "wasm_fileset.d.ts",
    ],
)

mediapipe_ts_library(
    name = "task_runner",
    srcs = ["task_runner.ts"],
    deps = [
        ":core",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/components/processors:base_options",
        "//mediapipe/web/graph_runner:graph_runner_image_lib_ts",
        "//mediapipe/web/graph_runner:graph_runner_ts",
        "//mediapipe/web/graph_runner:register_model_resources_graph_service_ts",
    ],
)

mediapipe_ts_library(
    name = "fileset_resolver",
    srcs = ["fileset_resolver.ts"],
    deps = [":core"],
)

mediapipe_ts_declaration(
    name = "classifier_options",
    srcs = ["classifier_options.d.ts"],
    deps = [":core"],
)

mediapipe_ts_declaration(
    name = "embedder_options",
    srcs = ["embedder_options.d.ts"],
    deps = [":core"],
)
