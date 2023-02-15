package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImageUtils",
    srcs = ["sources/MPPImage+Utils.mm"],
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
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
    ],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
)
