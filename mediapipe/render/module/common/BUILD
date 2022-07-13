load("//mediapipe/framework/tool:mediapipe_graph.bzl", "mediapipe_binary_graph")
package(default_visibility = ["//visibility:private"])
cc_library(
    name = "olamodule_common_library",
    srcs = ["OlaGraph.cpp"],
    hdrs = ["OlaGraph.hpp"],
    visibility = ["//mediapipe/framework:mediapipe_internal"],
    deps = [
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
        "//mediapipe/gpu:pixel_buffer_pool_util",
        "//mediapipe/util:cpu_util",
        "//mediapipe/render/core:core",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@google_toolbox_for_mac//:GTM_Defines",
    ] + select({
        "//mediapipe:ios": [
            "//mediapipe/objc:CFHolder",
            "//mediapipe/objc:Weakify",
            "//mediapipe/objc:util",
        ],
    }),
)