# This contains the MediaPipe Interactive Segmenter Task.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "interactive_segmenter",
    srcs = [
        "interactive_segmenter.ts",
        "interactive_segmenter_result.ts",
    ],
    deps = [
        ":interactive_segmenter_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:keypoint",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:mask",
        "//mediapipe/tasks/web/vision/core:types",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/util:color_jspb_proto",
        "//mediapipe/util:render_data_jspb_proto",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "interactive_segmenter_types",
    srcs = ["interactive_segmenter_options.d.ts"],
    deps = [
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

mediapipe_ts_library(
    name = "interactive_segmenter_test_lib",
    testonly = True,
    srcs = [
        "interactive_segmenter_test.ts",
    ],
    deps = [
        ":interactive_segmenter",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:mask",
        "//mediapipe/util:render_data_jspb_proto",
        "//mediapipe/web/graph_runner:graph_runner_image_lib_ts",
    ],
)

jasmine_node_test(
    name = "interactive_segmenter_test",
    tags = ["nomsan"],
    deps = [":interactive_segmenter_test_lib"],
)
