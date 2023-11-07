package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPPoseLandmarkerResultProtobufHelpers",
    srcs = ["sources/MPPPoseLandmarkerResult+ProtobufHelpers.mm"],
    hdrs = ["sources/MPPPoseLandmarkerResult+ProtobufHelpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/cc/components/containers/proto:landmarks_detection_result_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/test/vision/utils:parse_proto_utils",
        "//mediapipe/tasks/ios/vision/pose_landmarker:MPPPoseLandmarkerResult",
        "//mediapipe/tasks/ios/vision/pose_landmarker/utils:MPPPoseLandmarkerResultHelpers",
    ],
)
