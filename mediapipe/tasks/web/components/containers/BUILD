# This package contains options shared by all MediaPipe Tasks for Web.

load("//mediapipe/framework/port:build_config.bzl", "ts_declaration")

package(default_visibility = ["//mediapipe/tasks:internal"])

ts_declaration(
    name = "bounding_box",
    srcs = ["bounding_box.d.ts"],
)

ts_declaration(
    name = "category",
    srcs = ["category.d.ts"],
)

ts_declaration(
    name = "classification_result",
    srcs = ["classification_result.d.ts"],
    deps = [":category"],
)

ts_declaration(
    name = "detection_result",
    srcs = ["detection_result.d.ts"],
    deps = [
        ":bounding_box",
        ":category",
        ":keypoint",
    ],
)

ts_declaration(
    name = "keypoint",
    srcs = ["keypoint.d.ts"],
)

ts_declaration(
    name = "landmark",
    srcs = ["landmark.d.ts"],
)

ts_declaration(
    name = "embedding_result",
    srcs = ["embedding_result.d.ts"],
)

ts_declaration(
    name = "rect",
    srcs = ["rect.d.ts"],
)

ts_declaration(
    name = "matrix",
    srcs = ["matrix.d.ts"],
)
