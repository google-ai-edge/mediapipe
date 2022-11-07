# This contains the MediaPipe Object Detector Task.
#
# This task takes video frames and outputs synchronized frames along with
# the detection results for one or more object categories, using Object Detector.

load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "object_detector",
    srcs = [
        "object_detector.ts",
        "object_detector_options.d.ts",
        "object_detector_result.d.ts",
    ],
    deps = [
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:detection_jspb_proto",
        "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/processors:base_options",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner",
        "//mediapipe/web/graph_runner:wasm_mediapipe_lib_ts",
    ],
)
