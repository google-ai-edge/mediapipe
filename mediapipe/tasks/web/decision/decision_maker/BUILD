load("//mediapipe/framework/port:build_config.bzl", "ts_declaration", "ts_library")
load("//mediapipe/tasks/web:build_defs.bzl", "jasmine_node_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "decision_maker",
    srcs = ["decision_maker.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":decision_maker_model_types",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

ts_declaration(
    name = "decision_maker_model_types",
    srcs = [
        "decision_maker_options.d.ts",
        "decision_maker_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

ts_library(
    name = "decision_maker_test_lib",
    testonly = True,
    srcs = ["decision_maker_test.ts"],
    deps = [
        ":decision_maker",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

jasmine_node_test(
    name = "decision_maker_test",
    srcs = [":decision_maker_test_lib"],
    tags = ["nobuilder"],
)
