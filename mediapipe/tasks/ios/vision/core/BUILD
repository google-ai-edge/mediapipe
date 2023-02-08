package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImage",
    srcs = ["sources/MPPImage.m"],
    hdrs = ["sources/MPPImage.h"],
    module_name = "MPPImage",
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
    ],
    sdk_frameworks = [
        "CoreMedia",
        "CoreVideo",
        "UIKit",
    ],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
)
