load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_framework")

ios_framework(
    name = "OlaRenderDevelopFramework",
    hdrs = [
        "OlaRenderDevelop.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.olarender.develop",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
        ":OlaRenderDevelop",
    ],
)

objc_library(
    name = "OlaRenderDevelop",
    hdrs = [
        "OlaRenderDevelop.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/render/core:core",
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
    ],
)
