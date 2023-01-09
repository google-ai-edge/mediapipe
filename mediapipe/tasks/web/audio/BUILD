# This contains the MediaPipe Audio Tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

exports_files(["README.md"])

mediapipe_ts_library(
    name = "audio_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/audio/audio_classifier",
        "//mediapipe/tasks/web/audio/audio_embedder",
        "//mediapipe/tasks/web/core:fileset_resolver",
    ],
)
