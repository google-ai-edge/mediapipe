# This contains the MediaPipe Interactive Segmenter Task.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("//mediapipe/framework/port:build_config.bzl", "ts_library")
load("//mediapipe/tasks/web:build_defs.bzl", "jasmine_node_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "interactive_segmenter",
    srcs = [
        "interactive_segmenter.ts",
    ],
    deps = [
        ":interactive_segmenter_types",
        "//mediapipe/calculators/tensor:inference_calculator_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:acceleration_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/interactive_segmenter/proto:stroke_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/tasks/web/vision/core:image_shader_context",
        "//mediapipe/tasks/web/vision/core:mask",
        "//mediapipe/web/graph_runner:graph_runner_image_lib_ts",
        "//mediapipe/web/graph_runner:graph_runner_ts",
        "//mediapipe/web/graph_runner:platform_utils",
    ],
)

ts_library(
    name = "interactive_segmenter_types",
    srcs = ["interactive_segmenter_options.ts"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

ts_library(
    name = "interactive_segmenter_test_lib",
    testonly = True,
    srcs = [
        "interactive_segmenter_test.ts",
    ],
    deps = [
        ":interactive_segmenter",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:fileset_resolver",
        "//mediapipe/tasks/web/core:task_logger",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:drawing_utils",
        "//mediapipe/tasks/web/vision/core:image_shader_context",
        "//mediapipe/tasks/web/vision/core:mask",
        "//mediapipe/web/graph_runner:graph_runner_image_lib_ts",
        "//mediapipe/web/graph_runner:platform_utils",
    ],
)

# Headless unit tests verifying options contract, DOM mocking, and native Wasm memory bridge.
# Executes natively on standard Linux development clients in NodeJS using:
#
# bazel test //mediapipe/tasks/web/vision/interactive_segmenter:interactive_segmenter_test
jasmine_node_test(
    name = "interactive_segmenter_test",
    srcs = [":interactive_segmenter_test_lib"],
    tags = ["nomsan"],
)
