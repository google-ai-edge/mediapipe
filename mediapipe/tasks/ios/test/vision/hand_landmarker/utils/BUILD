package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPHandLandmarkerResultProtobufHelpers",
    srcs = ["sources/MPPHandLandmarkerResult+ProtobufHelpers.mm"],
    hdrs = ["sources/MPPHandLandmarkerResult+ProtobufHelpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/tasks/cc/components/containers/proto:landmarks_detection_result_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/test/vision/utils:parse_proto_utils",
        "//mediapipe/tasks/ios/vision/hand_landmarker:MPPHandLandmarkerResult",
        "//mediapipe/tasks/ios/vision/hand_landmarker/utils:MPPHandLandmarkerResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
