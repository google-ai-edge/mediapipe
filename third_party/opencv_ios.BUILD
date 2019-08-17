# Description:
#   OpenCV libraries for video/image processing on iOS

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

load(
    "@build_bazel_rules_apple//apple:apple.bzl",
    "apple_static_framework_import",
)

apple_static_framework_import(
    name = "OpencvFramework",
    framework_imports = glob(["opencv2.framework/**"]),
    visibility = ["//visibility:public"],
)

objc_library(
    name = "opencv_objc_lib",
    deps = [":OpencvFramework"],
)

cc_library(
    name = "opencv",
    hdrs = glob([
        "opencv2.framework/Versions/A/Headers/**/*.h*",
    ]),
    copts = [
        "-std=c++11",
        "-x objective-c++",
    ],
    include_prefix = "opencv2",
    linkopts = [
        "-framework AssetsLibrary",
        "-framework CoreFoundation",
        "-framework CoreGraphics",
        "-framework CoreMedia",
        "-framework Accelerate",
        "-framework CoreImage",
        "-framework AVFoundation",
        "-framework CoreVideo",
        "-framework QuartzCore",
    ],
    strip_include_prefix = "opencv2.framework/Versions/A/Headers",
    visibility = ["//visibility:public"],
    deps = [":opencv_objc_lib"],
)
