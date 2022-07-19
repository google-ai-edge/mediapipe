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
        "OlaRender.h",
        "OlaRenderIMP.h",
        ],
    # Use -Dverbose=-1 to turn off zlib's trace logging. (#3280)
    linkstatic = True,
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
    alwayslink = True,
)



objc_library(
    name = "OlaRenderLibrary",
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
    alwayslink = True,
)


ios_framework(
    name = "OlaRenderFramework",
    hdrs = [
        "image.h",
        "image_queue.h",
        "LockFreeQueue.h",
         "util.h", 
        "macros.h",
        "OlaRender.h",
        "OlaRenderIMP.h",
    ],
    # alwayslink=True,
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.dartrenderplugin",
    families = ["iphone", "ipad"],
    minimum_os_version = "10.0",
    deps = [
        "//mediapipe/modules/render_queue:OlaRenderLibrary",
    ],
)
