# This contains the MediaPipe Text Tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "text_lib",
    srcs = ["index.ts"],
    deps = [
        "//mediapipe/tasks/web/text/text_classifier",
        "//mediapipe/tasks/web/text/text_embedder",
    ],
)
