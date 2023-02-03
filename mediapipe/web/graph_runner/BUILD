# The TypeScript graph runner used by all MediaPipe Web tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

mediapipe_ts_library(
    name = "graph_runner_ts",
    srcs = [
        ":graph_runner.ts",
    ],
    allow_unoptimized_namespaces = True,
)

mediapipe_ts_library(
    name = "graph_runner_image_lib_ts",
    srcs = [
        ":graph_runner_image_lib.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [":graph_runner_ts"],
)

mediapipe_ts_library(
    name = "register_model_resources_graph_service_ts",
    srcs = [
        ":register_model_resources_graph_service.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [":graph_runner_ts"],
)
