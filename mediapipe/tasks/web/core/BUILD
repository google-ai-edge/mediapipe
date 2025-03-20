# This package contains options shared by all MediaPipe Tasks for Web.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_declaration(
    name = "core",
    srcs = [
        "task_runner_options.d.ts",
        "wasm_fileset.d.ts",
    ],
)

mediapipe_ts_library(
    name = "task_runner",
    srcs = ["task_runner.ts"],
    deps = [
        ":core",
        "//mediapipe/calculators/tensor:inference_calculator_jspb_proto",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:acceleration_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_jspb_proto",
        "//mediapipe/tasks/web/genai/llm_inference:model_loading_utils",
        "//mediapipe/web/graph_runner:graph_runner_ts",
        "//mediapipe/web/graph_runner:register_model_resources_graph_service_ts",
    ],
)

genrule(
    name = "run_script_genrule",
    srcs = ["fileset_resolver.ts.template"],
    outs = ["fileset_resolver.ts"],
    cmd = "cp $< $@",
)

mediapipe_ts_library(
    name = "fileset_resolver",
    srcs = ["fileset_resolver.ts"],
    visibility = ["//visibility:public"],
    deps = [":core"],
)

mediapipe_ts_library(
    name = "task_runner_test_utils",
    testonly = True,
    srcs = [
        "task_runner_test_utils.ts",
    ],
    deps = [
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/web/graph_runner:graph_runner_ts",
        "//mediapipe/web/graph_runner:register_model_resources_graph_service_ts",
    ],
)

mediapipe_ts_library(
    name = "task_runner_test_lib",
    testonly = True,
    srcs = [
        "task_runner_test.ts",
    ],
    deps = [
        ":core",
        ":task_runner",
        ":task_runner_test_utils",
        "//mediapipe/calculators/tensor:inference_calculator_jspb_proto",
        "//mediapipe/gpu:gpu_origin_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

jasmine_node_test(
    name = "task_runner_test",
    srcs = [":task_runner_test_lib"],
)

mediapipe_ts_declaration(
    name = "classifier_options",
    srcs = ["classifier_options.d.ts"],
    deps = [":core"],
)

mediapipe_ts_declaration(
    name = "embedder_options",
    srcs = ["embedder_options.d.ts"],
    deps = [":core"],
)
