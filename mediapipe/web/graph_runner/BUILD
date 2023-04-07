# The TypeScript graph runner used by all MediaPipe Web tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")
load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

mediapipe_ts_library(
    name = "graph_runner_ts",
    srcs = [
        ":graph_runner.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [":platform_utils"],
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

mediapipe_ts_library(
    name = "platform_utils",
    srcs = [
        "platform_utils.ts",
    ],
)

mediapipe_ts_library(
    name = "platform_utils_test_lib",
    testonly = True,
    srcs = [
        "platform_utils.test.ts",
    ],
    deps = [":platform_utils"],
)

jasmine_node_test(
    name = "platform_utils_test",
    deps = [
        ":platform_utils_test_lib",
    ],
)
