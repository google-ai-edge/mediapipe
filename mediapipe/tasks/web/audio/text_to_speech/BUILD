# This contains the MediaPipe TextToSpeech Web Task.

load("//mediapipe/framework/port:build_config.bzl", "ts_declaration", "ts_library")
load("//mediapipe/tasks/web:build_defs.bzl", "jasmine_node_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "text_to_speech",
    srcs = ["text_to_speech.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":text_to_speech_types",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

ts_declaration(
    name = "text_to_speech_types",
    srcs = [
        "text_to_speech_options.d.ts",
        "text_to_speech_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

ts_library(
    name = "text_to_speech_test_lib",
    testonly = True,
    srcs = ["text_to_speech_test.ts"],
    deps = [
        ":text_to_speech",
        "//mediapipe/tasks/web/core:task_logger",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

jasmine_node_test(
    name = "text_to_speech_test",
    srcs = [":text_to_speech_test_lib"],
)
