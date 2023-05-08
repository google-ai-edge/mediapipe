# This package contains options shared by all MediaPipe Vision Tasks for Web.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_declaration(
    name = "image_processing_options",
    srcs = ["image_processing_options.d.ts"],
    deps = [
        "//mediapipe/tasks/web/components/containers:rect",
    ],
)

mediapipe_ts_declaration(
    name = "vision_task_options",
    srcs = ["vision_task_options.d.ts"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

mediapipe_ts_declaration(
    name = "types",
    srcs = ["types.d.ts"],
    deps = [
        "//mediapipe/tasks/web/components/containers:keypoint",
    ],
)

mediapipe_ts_library(
    name = "drawing_utils",
    srcs = ["drawing_utils.ts"],
    deps = [
        ":types",
        "//mediapipe/tasks/web/components/containers:bounding_box",
        "//mediapipe/tasks/web/components/containers:landmark",
    ],
)

mediapipe_ts_library(
    name = "image",
    srcs = [
        "image.ts",
        "image_shader_context.ts",
    ],
)

mediapipe_ts_library(
    name = "image_test_lib",
    testonly = True,
    srcs = ["image.test.ts"],
    deps = [":image"],
)

jasmine_node_test(
    name = "image_test",
    deps = [":image_test_lib"],
)

mediapipe_ts_library(
    name = "mask",
    srcs = ["mask.ts"],
    deps = [":image"],
)

mediapipe_ts_library(
    name = "mask_test_lib",
    testonly = True,
    srcs = ["mask.test.ts"],
    deps = [
        ":image",
        ":mask",
    ],
)

jasmine_node_test(
    name = "mask_test",
    deps = [":mask_test_lib"],
)

mediapipe_ts_library(
    name = "vision_task_runner",
    srcs = ["vision_task_runner.ts"],
    deps = [
        ":image",
        ":image_processing_options",
        ":mask",
        ":vision_task_options",
        "//mediapipe/framework/formats:rect_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_image_lib_ts",
        "//mediapipe/web/graph_runner:graph_runner_ts",
        "//mediapipe/web/graph_runner:platform_utils",
        "//mediapipe/web/graph_runner:register_model_resources_graph_service_ts",
    ],
)

mediapipe_ts_library(
    name = "vision_task_runner_test_lib",
    testonly = True,
    srcs = ["vision_task_runner.test.ts"],
    deps = [
        ":image_processing_options",
        ":vision_task_options",
        ":vision_task_runner",
        "//mediapipe/framework/formats:rect_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_library(
    name = "render_utils",
    srcs = ["render_utils.ts"],
)

jasmine_node_test(
    name = "vision_task_runner_test",
    deps = [":vision_task_runner_test_lib"],
)
