package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImage",
    srcs = ["sources/MPPImage.m"],
    hdrs = ["sources/MPPImage.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    module_name = "MPPImage",
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:UIKit",
    ],
)

objc_library(
    name = "MPPRunningMode",
    hdrs = ["sources/MPPRunningMode.h"],
    module_name = "MPPRunningMode",
)

objc_library(
    name = "MPPVisionPacketCreator",
    srcs = ["sources/MPPVisionPacketCreator.mm"],
    hdrs = ["sources/MPPVisionPacketCreator.h"],
    sdk_frameworks = ["UIKit"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core/utils:MPPImageUtils",
        "//mediapipe/framework/formats:rect_cc_proto",
    ],
)

objc_library(
    name = "MPPVisionTaskRunner",
    srcs = ["sources/MPPVisionTaskRunner.mm"],
    hdrs = ["sources/MPPVisionTaskRunner.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        ":MPPRunningMode",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/core:MPPTaskRunner",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
    ],
)
