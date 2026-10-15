load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/port:build_config.bzl", "ts_declaration", "ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "text_proofreader",
    srcs = ["text_proofreader.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":text_proofreader_types",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

ts_declaration(
    name = "text_proofreader_types",
    srcs = [
        "text_proofreader_options.d.ts",
        "text_proofreader_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)
