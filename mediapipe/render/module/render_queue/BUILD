load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

licenses(["notice"])

package(
    default_visibility = ["//visibility:public"],
    features = ["no_layering_check"],
)

alias(
    name = "olarender_jni",
    actual = ":olarender",
)

cc_library(
    name = "olarender",
    srcs = [
        "util.cpp",
        "image_queue.cpp",
        "image.cpp",
        "OlaRender.cpp",
        "OlaRenderIMP.cpp",
    ]+ select({
        "//conditions:default": [],
        "//mediapipe:android": [
            "RenderJni.cpp",
        ],
    }),
    hdrs = [
        "util.h", 
        "macros.h",
        "image_queue.h",
        "image.h",
        "LockFreeQueue.h",
        "OlaRender.h",
        "OlaRenderIMP.h",
        ],
    linkopts = select({
        "//conditions:default": [
            "-lGLESv2",
            "-lEGL",
        ],
        "//mediapipe:android": [
            "-ljnigraphics",
            "-lGLESv3",
            "-lEGL",
            "-landroid",
            "-llog",
        ],
        "//mediapipe:ios": [
            "-framework OpenGLES",
            "-framework CoreVideo",
        ],
        "//mediapipe:macos": [
            "-framework OpenGL",
            "-framework CoreVideo",
        ],
    }),
    
    # Use -Dverbose=-1 to turn off zlib's trace logging. (#3280)
    # linkstatic = True,
    # linkopts = GL_BASE_LINK_OPTS_OSS,
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/render/module/render_queue/math:math",
    ],
    copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
         "//mediapipe:android": [
            "-std=c++11 -frtti -fexceptions -fpermissive -DVERSION_LIMIT",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)


objc_library(
    name = "OlaRenderLibrary",
    visibility = ["//visibility:public"],
    deps = [   
        "//mediapipe/render/module/render_queue:olarender",
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
        "//mediapipe/render/module/render_queue:OlaRenderLibrary",
    ],
)