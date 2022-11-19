# This contains the MediaPipe Audio Tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "audio_lib",
    srcs = ["index.ts"],
    deps = [
        "//mediapipe/tasks/web/audio/audio_classifier",
        "//mediapipe/tasks/web/audio/audio_embedder",
    ],
)
