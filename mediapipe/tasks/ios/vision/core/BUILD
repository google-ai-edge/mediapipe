package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImage",
    srcs = ["sources/MPPImage.h"],
    hdrs = ["sources/MPPImage.m"],
    module_name = "MPPImage",
    sdk_frameworks = [
        "CoreMedia",
        "CoreVideo",
        "UIKit",
    ],
    deps = [
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
    ],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
)
