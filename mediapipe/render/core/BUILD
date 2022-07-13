
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")
load("//mediapipe/framework:more_selects.bzl", "more_selects")

package(default_visibility = ["//visibility:public"])


GL_BASE_LINK_OPTS = select({
    "//conditions:default": [],
    "//mediapipe:android": [
        "-lGLESv3",
        "-lEGL",
        "-ljnigraphics",
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
        "-lGLESv3",
        "-lEGL",
        "-ljnigraphics",
    ],
    "//mediapipe:android": [],
    "//mediapipe:ios": [
            "-framework AVFoundation",
            "-framework CoreVideo",
            "-framework CoreGraphics",
            "-framework CoreMedia",
            "-framework GLKit",
            "-framework QuartzCore",],
})

cc_library(
    name = "gpuimageutil",
    srcs = ["GPUImageUtil.cpp"],
    hdrs = ["GPUImageUtil.h"],
    visibility = ["//path/to/package:__pkg__"],
    deps = select({
        "//mediapipe:android": [],
        "//mediapipe:apple": [],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "gpuimagemacros",
    hdrs = ["GPUImageMacros.h"],
    features = ["-layering_check"],

    linkopts = select({
        "//conditions:default": [],
        "//mediapipe:ios": [
            "-framework OpenGLES",
            "-framework CoreVideo",
            "-framework AVFoundation",
        ],
        "//mediapipe:macos": [
            "-framework OpenGL",
            "-framework CoreVideo",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [":gpuimageutil"] + select({
        "//mediapipe:android": [],
        "//mediapipe:apple": [],
        "//conditions:default": [],
    }),
)

cc_library(
    name="ref",
    srcs=["Ref.cpp"],
    hdrs=["Ref.hpp"],
    visibility = ["//visibility:public"],
    deps = [":gpuimagemacros"]
)

cc_library(
    name="gpuimagemath",
    srcs=["math.cpp"],
    hdrs=["math.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpuimagemacros",
    ],
)




cc_library(
    name = "core",
    srcs = [
        "FramebufferCache.cpp",
        "Framebuffer.cpp",
        "Target.cpp",
        "Context.cpp",
        "Filter.cpp",
        "GLProgram.cpp",
        "Source.cpp",
        "SourceImage.cpp",
        "IOSTarget.cpp",
        "SourceCamera.cpp",
        "TargetView.cpp",
        "FilterGroup.cpp"
        ],
    hdrs = [
        "FramebufferCache.hpp",
        "Framebuffer.hpp",
        "Target.hpp",
        "Filter.hpp",
        "Context.hpp",
        "GLProgram.hpp",
        "Source.hpp",
        "SourceImage.hpp",
        "CVFramebuffer.hpp",
        "GPUImageTarget.h",
        "IOSTarget.hpp",
        "GPUImage-x.h",
        "SourceCamera.hpp",
        "TargetView.hpp",
        "FilterGroup.hpp",
        ],
    visibility = ["//visibility:public"],
    deps = [
        ":gpuimageutil",
        ":gpuimagemacros",
        ":ref",
        "//mediapipe/render/core/math:math",
        ":gpuimagemath"
    ],
     copts = select({
        "//mediapipe:apple": [
            "-x objective-c++",
            "-fobjc-arc",  # enable reference-counting
        ],
        "//conditions:default": [],
    }),
)


