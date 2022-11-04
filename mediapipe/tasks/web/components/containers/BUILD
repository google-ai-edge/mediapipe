# This package contains options shared by all MediaPipe Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "category",
    srcs = ["category.d.ts"],
)

mediapipe_ts_library(
    name = "classifications",
    srcs = ["classifications.d.ts"],
    deps = [":category"],
)

mediapipe_ts_library(
    name = "landmark",
    srcs = ["landmark.d.ts"],
)
