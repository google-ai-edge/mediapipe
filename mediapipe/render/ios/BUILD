load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)


load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

ios_framework(
    name = "OlaRenderFramework",
    hdrs = [
        "OlaRender.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.noppelab.MyFramework",
    families = ["iphone", "ipad"],
    minimum_os_version = "10.0",
    deps = [
        ":OlaRender",
        "@ios_opencv//:OpencvFramework",
    ],
)

objc_library(
    name = "OlaRender",
    srcs = [
        "OlaRender.mm",
    ],
    hdrs = [
        "OlaRender.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/render/core:core",
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/objc:mediapipe_layer_renderer",
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
        "AssetsLibrary",
    ],
)
