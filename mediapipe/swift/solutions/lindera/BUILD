load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "lindera",
    srcs = ["Lindera.swift","Asensei3D.swift"],
    linkopts = [
        "-lc++",
        "-std=c++17",
        "-lstdc++",
    ],
    module_name = "LinderaDetection",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/objc/solutions/posetracking_gpu:posetracking_gpu_solution",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
    ] + select({
        "//mediapipe:ios_i386": [],
        "//mediapipe:ios_x86_64": [],
        "//conditions:default": [
            "//mediapipe/graphs/pose_tracking:pose_tracking_gpu_deps",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
)
