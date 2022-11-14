# This package contains options shared by all MediaPipe Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "core",
    srcs = [
        "base_options.d.ts",
        "wasm_loader_options.d.ts",
    ],
)

mediapipe_ts_library(
    name = "task_runner",
    srcs = [
        "task_runner.ts",
    ],
    deps = [
        "//mediapipe/web/graph_runner:register_model_resources_graph_service_ts",
        "//mediapipe/web/graph_runner:wasm_mediapipe_image_lib_ts",
        "//mediapipe/web/graph_runner:wasm_mediapipe_lib_ts",
    ],
)

mediapipe_ts_library(
    name = "classifier_options",
    srcs = [
        "classifier_options.d.ts",
    ],
    deps = [":core"],
)
