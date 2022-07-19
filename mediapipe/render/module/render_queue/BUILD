load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

GL_BASE_LINK_OPTS = select({
    "//conditions:default": [],
    "//mediapipe:android": [
        "-lGLESv3",
        "-lEGL",
        # Note: on Android, libGLESv3.so is normally a symlink to
        # libGLESv2.so, so we don't need to link to it. In fact, we
        # do not _want_ to link to it, or we would be unable to load
        # on API level < 18, where the symlink is missing entirely.
        # Note: if we ever find a strange version of Android where the
        # GLESv3 library is not a symlink, we will have to load it at
        # runtime. Weak GLESv3 symbols will still be resolved if we
        # load it early enough.
    ],
    "//mediapipe:ios": [
        "-framework OpenGLES",
        "-framework CoreVideo",
    ],
    "//mediapipe:macos": [
        "-framework OpenGL",
        "-framework CoreVideo",
    ],
})

# This is @unused internally.
GL_BASE_LINK_OPTS_OSS = GL_BASE_LINK_OPTS + select({
    "//conditions:default": [
        # Use GLES/EGL on linux.
        # Requires support from graphics card driver (nvidia,mesa,etc..)
        # and libraries to be installed.
        # Ex: libegl1-mesa-dev libgles2-mesa-dev, or libegl1-nvidia libgles2-nvidia, etc...
        "-lGLESv2",
        "-lEGL",
    ],
    "//mediapipe:android": [],
    "//mediapipe:ios": [],
})

cc_library(
    name = "gl_base",
    defines = select({
        "//mediapipe:apple": [
            "GLES_SILENCE_DEPRECATION=1",
        ],
        "//conditions:default": [],
    }),
    features = ["-layering_check"],
    linkopts = GL_BASE_LINK_OPTS_OSS,
    textual_hdrs = ["gl_base.h"],
    visibility = ["//visibility:public"],
    deps = [":gl_base_hdr"] + select({
        "//mediapipe:android": [],
        "//mediapipe:apple": [],
        "//conditions:default": [
        ],
    }),
)

cc_library(
    name = "gl_base_hdr",
    hdrs = ["gl_base.h"],
    features = ["-layering_check"],
    # Note: need the frameworks on Apple platforms to get the headers.
    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:ios": [
            "-framework OpenGLES",
            "-framework CoreVideo",
        ],
        "//mediapipe:macos": [
            "-framework OpenGL",
            "-framework CoreVideo",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "//mediapipe:android": [],
        "//mediapipe:apple": [],
        "//conditions:default": [
        ],
    }),
)


cc_library(
    name = "olarender",
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