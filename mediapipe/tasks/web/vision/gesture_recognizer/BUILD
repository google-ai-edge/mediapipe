# This contains the MediaPipe Gesture Recognizer Task.
#
# This task takes video frames and outputs synchronized frames along with
# the detection results for one or more gesture categories, using Gesture Recognizer.

load("@npm//@bazel/jasmine:index.bzl", "jasmine_node_test")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_ts_declaration", "mediapipe_ts_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

mediapipe_ts_library(
    name = "gesture_recognizer",
    srcs = ["gesture_recognizer.ts"],
    visibility = ["//visibility:public"],
    deps = [
        ":gesture_recognizer_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework:calculator_options_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/framework/formats:landmark_jspb_proto",
        "//mediapipe/tasks/cc/core/proto:base_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_classifier_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_recognizer_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:hand_gesture_recognizer_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_jspb_proto",
        "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_jspb_proto",
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/tasks/web/components/processors:classifier_options",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/vision/core:image_processing_options",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
        "//mediapipe/tasks/web/vision/hand_landmarker:hand_landmarks_connections",
        "//mediapipe/web/graph_runner:graph_runner_ts",
    ],
)

mediapipe_ts_declaration(
    name = "gesture_recognizer_types",
    srcs = [
        "gesture_recognizer_options.d.ts",
        "gesture_recognizer_result.d.ts",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/tasks/web/components/containers:category",
        "//mediapipe/tasks/web/components/containers:landmark",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:classifier_options",
        "//mediapipe/tasks/web/vision/core:vision_task_options",
    ],
)

mediapipe_ts_library(
    name = "gesture_recognizer_test_lib",
    testonly = True,
    srcs = [
        "gesture_recognizer_test.ts",
    ],
    deps = [
        ":gesture_recognizer",
        ":gesture_recognizer_types",
        "//mediapipe/framework:calculator_jspb_proto",
        "//mediapipe/framework/formats:classification_jspb_proto",
        "//mediapipe/framework/formats:landmark_jspb_proto",
        "//mediapipe/tasks/web/core",
        "//mediapipe/tasks/web/core:task_runner_test_utils",
        "//mediapipe/tasks/web/vision/core:vision_task_runner",
    ],
)

jasmine_node_test(
    name = "gesture_recognizer_test",
    srcs = [":gesture_recognizer_test_lib"],
    tags = ["nomsan"],
)
