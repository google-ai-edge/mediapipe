# This package contains options shared by all MediaPipe Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "classifier_options",
    srcs = ["classifier_options.ts"],
    deps = [
        "//mediapipe/tasks/cc/components/processors/proto:classifier_options_jspb_proto",
        "//mediapipe/tasks/web/core:classifier_options",
    ],
)

mediapipe_ts_library(
    name = "classifier_result",
    srcs = ["classifier_result.ts"],
    deps = [
        "//mediapipe/tasks/cc/components/containers/proto:classifications_jspb_proto",
        "//mediapipe/tasks/web/components/containers:classifications",
    ],
)

mediapipe_ts_library(
    name = "base_options",
    srcs = ["base_options.ts"],
    deps = [
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:external_file_jspb_proto",
        "//mediapipe/tasks/web/core",
    ],
)
