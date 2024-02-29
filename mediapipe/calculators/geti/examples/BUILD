licenses(["notice"])

package(default_visibility = ["//visibility:public"])
load("@mediapipe//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_simple_subgraph")

cc_library(
    name = "demo_dependencies",
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "//mediapipe/calculators/geti/inference:inference_calculators",
        "//mediapipe/calculators/geti/utils:utils",
        "//mediapipe/calculators/geti/serialization:calculators"
    ]
)

cc_binary(
    name = "anomaly_calculator_demo",
    srcs = ["anomaly_calculator_demo.cc"],
    data = [
        "//mediapipe/calculators/geti/graphs:graphs",
    ],
    deps = [
        ":demo_dependencies",
    ],
)
