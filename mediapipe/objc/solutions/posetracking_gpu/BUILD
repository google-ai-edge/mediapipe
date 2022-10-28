load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_framework", "ios_static_framework", "ios_unit_test")

ios_static_framework(
    name = "MPPoseTracking",
    #    hdrs = EDGE_ENGINE_HDRS,
    bundle_name = "MPPoseTracking",
    minimum_os_version = "12.0",
    visibility = ["//visibility:public"],
    deps = [
        ":posetracking_gpu_solution",
        #        "//third_party:opencv",
        #        "@ios_opencv//:OpencvFramework",
    ],
)

objc_library(
    name = "posetracking_gpu_solution",
    srcs = glob([
        "*.h",
        "*.mm",
    ]),
    hdrs = [
        "PoseTracking.h",
        "PoseTrackingOptions.h",
        "PoseTrackingResults.h",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    data = [
        "//mediapipe/graphs/pose_tracking:pose_tracking_gpu.binarypb",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_heavy.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_lite.tflite",
    ],
    module_name = "MPPoseTracking",
    sdk_frameworks = ["Accelerate"],
    visibility = ["//visibility:public"],
    deps = [
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
    alwayslink = True,
)
