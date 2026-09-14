load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/port:build_config.bzl", "ts_declaration", "ts_library")
load("//mediapipe/tasks/web:build_defs.bzl", "jasmine_node_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "text_summarizer",
    srcs = ["text_summarizer.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":text_summarizer_types",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

ts_declaration(
    name = "text_summarizer_types",
    srcs = [
        "text_summarizer_options.d.ts",
        "text_summarizer_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

ts_library(
    name = "text_summarizer_test_lib",
    testonly = True,
    srcs = ["text_summarizer_test.ts"],
    deps = [
        ":text_summarizer",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

jasmine_node_test(
    name = "text_summarizer_test",
    srcs = [":text_summarizer_test_lib"],
)
