load("//mediapipe/gpu:metal.bzl", "metal_library")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")


ios_framework(
    name = "OlaCameraFramework",
    hdrs = glob(["*.h"]),
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.cameraframework",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
        ":OlaCamera",
    ],
)

objc_library(
    name = "OlaCamera",
    srcs = glob(["*.m","*.mm"]),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    deps = [   
        ":OlaCameraMetalLibrary",
    ],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    sdk_frameworks = [
        "AVFoundation",
        "CoreGraphics",
        "CoreMedia",
        "UIKit",
        "OpenGLES",
        "MetalKit",
        "GLKit",
        "Metal",
        "CoreVideo",
        "IOSurface",
        "Foundation",
    ],
)

metal_library(
    name = "OlaCameraMetalLibrary",
    srcs = ["OlaCameraFramework.metal"],
)