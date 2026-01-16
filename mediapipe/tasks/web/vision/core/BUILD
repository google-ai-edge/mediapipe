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

mediapipe_ts_library(
    name = "types",
    srcs = ["types.ts"],
    deps = [
        "//mediapipe/tasks/web/components/containers:keypoint",
    ],
)

mediapipe_ts_library(
    name = "drawing_utils",
    srcs = [
        "drawing_utils.ts",
        "drawing_utils_category_mask.ts",
        "drawing_utils_confidence_mask.ts",
    ],
    deps = [
        ":image",
        ":image_shader_context",
        ":mask",
        ":types",
        "//mediapipe/tasks/web/components/containers:bounding_box",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_library(
    name = "drawing_utils_test_lib",
    testonly = True,
    srcs = ["drawing_utils.test.ts"],
    deps = [
        ":drawing_utils",
        ":image",
        ":image_shader_context",
        ":mask",
    ],
)

jasmine_node_test(
    name = "drawing_utils_test",
    srcs = [":drawing_utils_test_lib"],
)

mediapipe_ts_library(
    name = "image",
    srcs = ["image.ts"],
    deps = ["image_shader_context"],
)

mediapipe_ts_library(
    name = "image_shader_context",
    srcs = ["image_shader_context.ts"],
)

mediapipe_ts_library(
    name = "image_test_lib",
    testonly = True,
    srcs = ["image.test.ts"],
    deps = [
        ":image",
        ":image_shader_context",
    ],
)

jasmine_node_test(
    name = "image_test",
    srcs = [":image_test_lib"],
)

mediapipe_ts_library(
    name = "mask",
    srcs = ["mask.ts"],
    deps = [
        ":image",
        ":image_shader_context",
        "//mediapipe/web/graph_runner:platform_utils",
    ],
)

mediapipe_ts_library(
    name = "mask_test_lib",
    testonly = True,
    srcs = ["mask.test.ts"],
    deps = [
        ":image",
        ":image_shader_context",
        ":mask",
    ],
)

jasmine_node_test(
    name = "mask_test",
    srcs = [":mask_test_lib"],
)

mediapipe_ts_library(
    name = "vision_task_runner",
    srcs = ["vision_task_runner.ts"],
    deps = [
        ":image",
        ":image_processing_options",
        ":image_shader_context",
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
    srcs = ["vision_task_runner_test.ts"],
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

jasmine_node_test(
    name = "vision_task_runner_test",
    srcs = [":vision_task_runner_test_lib"],
)
