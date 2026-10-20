# This contains the MediaPipe AudioTranscriber Web Task.

load("//mediapipe/framework/port:build_config.bzl", "ts_declaration", "ts_library")
load("//mediapipe/tasks/web:build_defs.bzl", "jasmine_node_test")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

ts_library(
    name = "audio_transcriber",
    srcs = ["audio_transcriber.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":audio_transcriber_types",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

ts_declaration(
    name = "audio_transcriber_types",
    srcs = [
        "audio_transcriber_options.d.ts",
        "audio_transcriber_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

ts_library(
    name = "audio_transcriber_test_lib",
    testonly = True,
    srcs = ["audio_transcriber_test.ts"],
    deps = [
        ":audio_transcriber",
        "//mediapipe/tasks/web/core:task_logger",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

jasmine_node_test(
    name = "audio_transcriber_test",
    srcs = [":audio_transcriber_test_lib"],
)
