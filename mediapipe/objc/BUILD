load("//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_binary_graph")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

cc_library(
    name = "CFHolder",
    # Header is excluded on non-ios so you can still build :all.
    hdrs = select({
        "//mediapipe:apple": ["CFHolder.h"],
        "//conditions:default": [],
    }),
    features = ["layering_check"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    features = [
        "-parse_headers",
        "layering_check",
    ],
    linkopts = [
        "-framework Accelerate",
        "-framework CoreFoundation",
        "-framework CoreGraphics",
        "-framework CoreVideo",
    ],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":CFHolder",
        "//mediapipe/framework:packet",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/memory",
    ],
)

objc_library(
    name = "Weakify",
    hdrs = ["Weakify.h"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
)

MEDIAPIPE_IOS_SRCS = [
    "MPPGraph.cc",
    "MPPTimestampConverter.cc",
    "NSError+util_status.cc",
]

MEDIAPIPE_IOS_HDRS = [
    "MPPGraph.h",
    "MPPTimestampConverter.h",
    "NSError+util_status.h",
]

cc_library(
    name = "mediapipe_framework_ios",
    srcs = MEDIAPIPE_IOS_SRCS,
    hdrs = MEDIAPIPE_IOS_HDRS,
    copts = [
        "-x objective-c++",
        "-fobjc-arc",  # enable reference-counting
        "-Wno-shorten-64-to-32",
    ],
    # This build rule is public to allow external customers to build their own iOS apps.
    visibility = ["//visibility:public"],
    deps = [
        ":CFHolder",
        ":util",
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework:mediapipe_profiling",
        "//mediapipe/framework/formats:image",
        "//mediapipe/framework/port:map_util",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:source_location",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:statusor",
        "//mediapipe/framework/port:threadpool",
        "//mediapipe/gpu:gl_base",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/gpu:graph_support",
        "//mediapipe/gpu:metal_shared_resources",
        "//mediapipe/gpu:pixel_buffer_pool_util",
        "//mediapipe/util:cpu_util",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Accelerate",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@google_toolbox_for_mac//:GTM_Defines",
    ],
    alwayslink = 1,
)

objc_library(
    name = "mediapipe_input_sources_ios",
    srcs = [
        "MPPInputSource.m",
        "MPPPlayerInputSource.m",
    ] + select({
        "//mediapipe:ios": [
            "MPPCameraInputSource.m",
            "MPPDisplayLinkWeakTarget.m",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "MPPInputSource.h",
        "MPPPlayerInputSource.h",
    ] + select({
        "//mediapipe:ios": [
            "MPPCameraInputSource.h",
            "MPPDisplayLinkWeakTarget.h",
        ],
        "//conditions:default": [],
    }),
    # This build rule is public to allow external customers to build their own iOS apps.
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:CoreAudio",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Foundation",
        "//third_party/apple_frameworks:MediaToolbox",
    ],
)

objc_library(
    name = "mediapipe_gl_view_renderer",
    srcs = [
        "MPPGLViewRenderer.mm",
    ],
    hdrs = [
        "MPPGLViewRenderer.h",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":mediapipe_framework_ios",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gl_quad_renderer",
        "//mediapipe/gpu:gl_simple_shaders",
        "//third_party/apple_frameworks:Foundation",
        "//third_party/apple_frameworks:GLKit",
    ],
)

objc_library(
    name = "mediapipe_layer_renderer",
    srcs = [
        "MPPLayerRenderer.m",
    ],
    hdrs = [
        "MPPLayerRenderer.h",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    # This build rule is public to allow external customers to build their own iOS apps.
    visibility = ["//visibility:public"],
    deps = [
        ":mediapipe_framework_ios",
        ":mediapipe_gl_view_renderer",
        "//mediapipe/gpu:gl_calculator_helper",
        "//third_party/apple_frameworks:Foundation",
        "//third_party/apple_frameworks:GLKit",
    ],
)

objc_library(
    name = "CGImageRefUtils",
    srcs = [
        "CGImageRefUtils.mm",
    ],
    hdrs = [
        "CGImageRefUtils.h",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":mediapipe_framework_ios",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Foundation",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "mediapipe_audio_util",
    srcs = ["MediaPipeAudioUtil.mm"],
    hdrs = ["MediaPipeAudioUtil.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/framework/port:statusor",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:CoreAudio",
        "//third_party/apple_frameworks:CoreMedia",
        "@eigen//:eigen3",
    ],
)

objc_library(
    name = "MPPGraphTestBase",
    testonly = True,
    srcs = [
        "MPPGraphTestBase.mm",
    ],
    hdrs = [
        "MPPGraphTestBase.h",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":CGImageRefUtils",
        ":Weakify",
        ":mediapipe_framework_ios",
        "//mediapipe/framework:calculator_framework",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Accelerate",
        "//third_party/apple_frameworks:CoreGraphics",
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:GLKit",
        "//third_party/apple_frameworks:OpenGLES",
        "//third_party/apple_frameworks:QuartzCore",
        "//third_party/apple_frameworks:UIKit",
    ],
)

objc_library(
    name = "mediapipe_framework_ios_testLib",
    testonly = True,
    srcs = [
        "CFHolderTests.mm",
        "MPPDisplayLinkWeakTargetTests.mm",
        "MPPGraphTests.mm",
        "MediaPipeAudioUtilTests.mm",
    ],
    copts = [
        "-Wno-shorten-64-to-32",
    ],
    data = [
        "testdata/googlelogo_color_272x92dp.png",
    ],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
        ":CGImageRefUtils",
        ":MPPGraphTestBase",
        ":Weakify",
        ":mediapipe_audio_util",
        ":mediapipe_framework_ios",
        ":mediapipe_input_sources_ios",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Accelerate",
        "//third_party/apple_frameworks:CoreAudio",
        "//third_party/apple_frameworks:CoreGraphics",
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:GLKit",
        "//third_party/apple_frameworks:QuartzCore",
        "//third_party/apple_frameworks:UIKit",
    ],
)

[
    mediapipe_binary_graph(
        name = graph.split("/")[-1].rsplit(".", 1)[0] + "_graph",
        graph = graph,
        output_name = "%s.binarypb" % graph.split("/")[-1].rsplit(".", 1)[0],
        visibility = ["//mediapipe/framework:mediapipe_internal"],
    )
    for graph in glob(["testdata/*.pbtxt"])
]

exports_files(
    [
        "testdata/googlelogo_color_272x92dp.png",
        "testdata/googlelogo_color_272x92dp_luminance.png",
        "testdata/sergey.png",
    ],
    visibility = [
        "//mediapipe:__subpackages__",
    ],
)
