# This contains the MediaPipe RAG Pipeline

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

licenses(["notice"])

mediapipe_ts_library(
    name = "rag_pipeline",
    srcs = ["rag_pipeline.ts"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/genai/llm_inference",
        "//mediapipe/web/graph_runner:graph_runner_ts",
        "//mediapipe/web/graph_runner:graph_runner_wasm_file_reference_ts",
    ],
)
