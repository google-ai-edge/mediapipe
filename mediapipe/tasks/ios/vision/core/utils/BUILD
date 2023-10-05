package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImageUtils",
    srcs = ["sources/MPPImage+Utils.mm"],
    hdrs = ["sources/MPPImage+Utils.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    module_name = "MPPImageUtils",
    deps = [
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//third_party/apple_frameworks:Accelerate",
        "//third_party/apple_frameworks:CoreGraphics",
        "//third_party/apple_frameworks:CoreImage",
        "//third_party/apple_frameworks:CoreVideo",
    ],
)
