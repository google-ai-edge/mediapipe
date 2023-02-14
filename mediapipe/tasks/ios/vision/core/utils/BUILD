package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImageUtils",
    srcs = ["sources/MPPImage+Utils.m"],
    hdrs = ["sources/MPPImage+Utils.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    module_name = "MPPImageUtils",
    sdk_frameworks = [
        "Accelerate",
        "CoreGraphics",
        "CoreImage",
        "CoreVideo",
        "UIKit",
    ],
    deps = [
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//third_party/apple_frameworks:UIKit",
    ],
)
