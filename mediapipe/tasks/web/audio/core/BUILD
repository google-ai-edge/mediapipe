# This package contains options shared by all MediaPipe Audio Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_declaration(
    name = "audio_task_options",
    srcs = ["audio_task_options.d.ts"],
    deps = [
        "//mediapipe/tasks/web/core",
    ],
)

mediapipe_ts_library(
    name = "audio_task_runner",
    srcs = ["audio_task_runner.ts"],
    deps = [
        ":audio_task_options",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/web/components/processors:base_options",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
    ],
)
