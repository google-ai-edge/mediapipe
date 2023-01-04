# This package contains options shared by all MediaPipe Vision Tasks for Web.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_declaration(
    name = "vision_task_options",
    srcs = ["vision_task_options.d.ts"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

mediapipe_ts_library(
    name = "vision_task_runner",
    srcs = ["vision_task_runner.ts"],
    deps = [
        ":vision_task_options",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_library(
    name = "vision_task_runner_test_lib",
    testonly = True,
    srcs = ["vision_task_runner.test.ts"],
    deps = [
        ":vision_task_options",
        ":vision_task_runner",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

jasmine_node_test(
    name = "vision_task_runner_test",
    deps = [":vision_task_runner_test_lib"],
)
