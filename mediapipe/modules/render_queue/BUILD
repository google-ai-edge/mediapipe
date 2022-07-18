load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

cc_library(
    name = "ios_olarender",
    srcs = [
        "util.cpp",
        "image_queue.cpp",
        "image.cpp",
        "OlaRender.cpp",
        "OlaRenderIMP.cpp",
        ],
    hdrs = [
        "util.h", 
        "macros.h",
        "image_queue.h",
        "image.h",
        "LockFreeQueue.h",
        "OlaRender.hpp",
        "OlaRenderIMP.hpp",
        ],
    # Use -Dverbose=-1 to turn off zlib's trace logging. (#3280)
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/modules/render_queue/math:math",
    ],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
)



objc_library(
    name = "OlaRenderLibrary",
    srcs = [
            "OlaRenderManager.mm",
            "image_queue.cpp",
            "image.cpp",
            ],
    hdrs = [
            "OlaRenderManager.h",
            "image_queue.h",
            "image.h",
            "LockFreeQueue.h",
            ],
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
    hdrs = [
        "OlaRenderManager.h",
        "image.h",
        "image_queue.h",
        "LockFreeQueue.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.dartrenderplugin",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
        "//mediapipe/modules/render_queue:OlaRenderLibrary",
    ],
)
