# This package contains options shared by all MediaPipe Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "category",
    srcs = ["category.d.ts"],
)

mediapipe_ts_library(
    name = "classification_result",
    srcs = ["classification_result.d.ts"],
    deps = [":category"],
)

mediapipe_ts_library(
    name = "landmark",
    srcs = ["landmark.d.ts"],
)

mediapipe_ts_library(
    name = "embedding_result",
    srcs = ["embedding_result.d.ts"],
)
