# This contains the MediaPipe Vision Tasks.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

mediapipe_ts_library(
    name = "vision_lib",
    srcs = ["index.ts"],
    deps = [
        "//mediapipe/tasks/web/vision/gesture_recognizer",
        "//mediapipe/tasks/web/vision/hand_landmarker",
        "//mediapipe/tasks/web/vision/image_classifier",
        "//mediapipe/tasks/web/vision/image_embedder",
        "//mediapipe/tasks/web/vision/object_detector",
    ],
)
