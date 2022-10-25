objc_library(
    name = "posetracking_gpu_solution",
    srcs = glob([
        "*.h",
        "*.mm",
    ]),
    module_name = "MPPoseTracking",
    hdrs = [
        "PoseTracking.h",
        "PoseTrackingOptions.h",
        "PoseTrackingResults.h"
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    data =  [
        "//mediapipe/graphs/pose_tracking:pose_tracking_gpu.binarypb",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_heavy.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_lite.tflite",    ] ,
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
)
