load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")
ios_framework(
    name = "OlaFaceUnityFramework",
    hdrs = [
        "OlaFaceUnity.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.olarender.develop",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
         ":OlaFaceUnityLibrary",
    ],
)

objc_library(
    name = "OlaFaceUnityLibrary",
    hdrs = ["OlaFaceUnity.h"],
    srcs = ["OlaFaceUnity.mm"],
    visibility = ["//visibility:public"],
    deps = [
         "//mediapipe/render/module/beauty:FaceMeshGPULibrary",
         "@ios_opencv//:OpencvFramework",
    ],
     copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
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
        "AssetsLibrary",
        "QuartzCore",
    ],
)
