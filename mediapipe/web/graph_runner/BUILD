# The TypeScript graph runner used by all MediaPipe Web tasks.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

# Public types for GraphRunner listeners
mediapipe_ts_declaration(
    name = "listener_types",
    srcs = [
        "listener_types.d.ts",
    ],
    visibility = ["//visibility:private"],
)

# Public API for GraphRunner
mediapipe_ts_declaration(
    name = "graph_runner_api",
    srcs = [
        "graph_runner_api.d.ts",
    ],
    deps = [
        ":listener_types",
        ":wasm_module_private",
    ],
)

# Public API for GraphRunner creation factory functions
mediapipe_ts_library(
    name = "graph_runner_factory_api",
    srcs = [
        "graph_runner_factory_api.d.ts",
    ],
    deps = [
        ":graph_runner_api",
        ":wasm_module_private",
    ],
)

# Private declarations for the Wasm module
mediapipe_ts_declaration(
    name = "wasm_module_private",
    srcs = ["wasm_module.d.ts"],
    visibility = ["//visibility:private"],
    deps = [":listener_types"],
)

# Internal GraphRunner implementation
mediapipe_ts_library(
    name = "graph_runner_ts",
    srcs = [
        ":graph_runner.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [
        ":graph_runner_api",
        ":graph_runner_factory_api",
        ":listener_types",
        ":platform_utils",
        ":run_script_helper",
        ":wasm_module_private",
    ],
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

genrule(
    name = "run_script_genrule",
    srcs = ["run_script_helper.ts.template"],
    outs = ["run_script_helper.ts"],
    cmd = "cp $< $@",
)

mediapipe_ts_library(
    name = "run_script_helper",
    srcs = ["run_script_helper.ts"],
)

jasmine_node_test(
    name = "platform_utils_test",
    srcs = [
        ":platform_utils_test_lib",
    ],
)

mediapipe_ts_library(
    name = "graph_runner_webgpu_ts",
    srcs = [
        "graph_runner_webgpu.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [
        ":graph_runner_ts",
    ],
)

mediapipe_ts_library(
    name = "graph_runner_wasm_file_reference_ts",
    srcs = [
        "graph_runner_wasm_file_reference.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [
        ":graph_runner_ts",
    ],
)

mediapipe_ts_library(
    name = "graph_runner_streaming_reader_ts",
    srcs = [
        "graph_runner_streaming_reader.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [
        ":graph_runner_ts",
    ],
)

mediapipe_ts_library(
    name = "graph_runner_llm_inference_lib_ts",
    srcs = [
        "graph_runner_llm_inference_lib.ts",
    ],
    allow_unoptimized_namespaces = True,
    deps = [
        ":graph_runner_streaming_reader_ts",
        ":graph_runner_ts",
        "//mediapipe/tasks/cc/genai/inference/proto:sampler_params_jspb_proto",
        "//mediapipe/tasks/web/genai/llm_inference:llm_inference_types",
        "//mediapipe/tasks/web/genai/llm_inference:model_loading_utils",
        "//mediapipe/tasks/web/genai/llm_inference/proto:llm_inference_graph_options_jspb_proto",
    ],
)
