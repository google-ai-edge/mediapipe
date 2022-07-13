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
        "OlaRenderPlayground.h",
    ],
    infoplists = ["Info.plist"],
    bundle_id = "com.ola.olarender.develop",
    families = ["iphone", "ipad"],
    minimum_os_version = "11.0",
    deps = [
         ":OlaRenderPlayground",
    ],
)

objc_library(
    name = "OlaRenderPlayground",
    hdrs = ["OlaRenderPlayground.h"],
    srcs = ["OlaRenderPlayground.mm"],
    visibility = ["//visibility:public"],
    deps = [
         "//mediapipe/render/core:core-ios",
    ]
)
