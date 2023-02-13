package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImageUtils",
    srcs = ["sources/MPPImage+Utils.m"],
    hdrs = ["sources/MPPImage+Utils.h"],
    module_name = "MPPImageUtils",
    sdk_frameworks = [
        "Accelerate",
        "CoreGraphics",
        "CoreImage",
        "CoreVideo",
    ],
    deps = [
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
    ],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
)