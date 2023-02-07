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
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:UIKit",
    ],
)
