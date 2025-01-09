# This contains the MediaPipe LLM Inference Task.
#
# This task takes text input and performs text generation
#

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "llm_inference",
    srcs = ["llm_inference.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":llm_inference_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/genai/inference/calculators:detokenizer_calculator_jspb_proto",
        "//mediapipe/tasks/cc/genai/inference/calculators:llm_gpu_calculator_jspb_proto",
        "//mediapipe/tasks/cc/genai/inference/calculators:model_data_calculator_jspb_proto",
        "//mediapipe/tasks/cc/genai/inference/calculators:tokenizer_calculator_jspb_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:llm_params_jspb_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:sampler_params_jspb_proto",
        "//mediapipe/tasks/cc/genai/inference/proto:transformer_params_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/tasks/web/genai/llm_inference/proto:llm_inference_graph_options_jspb_proto",
        "//mediapipe/web/graph_runner:graph_runner_streaming_reader_ts",
        "//mediapipe/web/graph_runner:graph_runner_ts",
        "//mediapipe/web/graph_runner:graph_runner_wasm_file_reference_ts",
        "//mediapipe/web/graph_runner:graph_runner_webgpu_ts",
    ],
)

mediapipe_ts_declaration(
    name = "llm_inference_types",
    srcs = [
        "llm_inference_options.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)
