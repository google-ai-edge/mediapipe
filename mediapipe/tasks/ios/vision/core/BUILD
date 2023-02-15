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
    sdk_frameworks = [
        "CoreMedia",
        "CoreVideo",
        "UIKit",
    ],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
    ],
)

objc_library(
    name = "MPPRunningMode",
    hdrs = ["sources/MPPRunningMode.h"],
    module_name = "MPPRunningMode",
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
        "//mediapipe/tasks/ios/core:MPPTaskRunner",
        "//mediapipe/tasks/ios/core:MPPVisionPacketCreator",
    ],
)

objc_library(
    name = "MPPVisionPacketCreator",
    srcs = ["sources/MPPVisionPacketCreator.mm"],
    hdrs = ["sources/MPPVisionPacketCreator.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:image",
        "//mediapipe/tasks/ios/vision/core/utils:MPPImageUtils",
    ],
)
