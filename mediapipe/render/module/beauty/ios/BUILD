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
         "//mediapipe/render/core:core-ios",
         "//mediapipe/render/module/beauty:FaceMeshGPULibrary",
         "@ios_opencv//:OpencvFramework",
    ]
)
