# This package contains options shared by all MediaPipe Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "core",
    srcs = [
        "base_options.d.ts",
        "wasm_loader_options.d.ts",
    ],
)

mediapipe_ts_library(
    name = "classifier_options",
    srcs = [
        "classifier_options.d.ts",
    ],
    deps = [":core"],
)
