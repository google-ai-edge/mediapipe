# This contains the MediaPipe Text Tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

exports_files(["README.md"])

mediapipe_ts_library(
    name = "text_lib",
    srcs = ["index.ts"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/core:fileset_resolver",
        "//mediapipe/tasks/web/text/text_classifier",
        "//mediapipe/tasks/web/text/text_embedder",
    ],
)
