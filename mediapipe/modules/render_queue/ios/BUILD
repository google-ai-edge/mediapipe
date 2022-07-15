load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

objc_library(
    name = "OlaRenderLibrary",
    srcs = ["OlaRenderManager.mm"],
    hdrs = ["OlaRenderManager.h"],
    visibility = ["//visibility:public"],
    deps = [   
        "//mediapipe/modules/render_queue:ios_olarender",
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
        "OpenGLES",
        "CoreVideo",
        "IOSurface",
        "Foundation",
        "UIKit",
    ],
)


ios_framework(
    name = "OlaRenderFramework",
    hdrs = ["OlaRenderManager.h"],
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.dartrenderplugin",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
        "//mediapipe/modules/render_queue/ios:OlaRenderLibrary",
    ],
)
