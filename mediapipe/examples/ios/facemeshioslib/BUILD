load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_framework"
)
load(
    "//mediapipe/examples/ios:bundle_id.bzl",
    "BUNDLE_ID_PREFIX",
    "example_provisioning",
)

licenses(["notice"])  # Apache 2.0

MIN_IOS_VERSION = "11.1"

IOS_FAMILIES = [
    "iphone",
    "ipad",
]

FRAMEWORK_HEADERS = [
    "FaceMesh.h",
]

ios_framework(
    name = "FaceMeshSDK",
    hdrs = FRAMEWORK_HEADERS,
    bundle_id = BUNDLE_ID_PREFIX + ".FaceMeshSDK",
    bundle_name = "FaceMeshSDK",
    families = IOS_FAMILIES,
    infoplists = [
        "//mediapipe/examples/ios/common:Info.plist",
#        "Info.plist",
    ],
    minimum_os_version = MIN_IOS_VERSION,
    visibility = ["//visibility:public"],
    deps = [
        ":FaceMeshObj",
        "@ios_opencv//:OpencvFramework",
    ],
)


objc_library(
    name = "FaceMeshObj",
    srcs = [
        "FaceMesh.mm",
    ],
    hdrs = FRAMEWORK_HEADERS,
    copts = ["-std=c++17"],
    data = [
        "//mediapipe/graphs/face_mesh:face_mesh_ios_lib_gpu.binarypb",
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    deps = [
        "//mediapipe/objc:mediapipe_framework_ios",
        "//mediapipe/objc:mediapipe_input_sources_ios",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:opencv_imgcodecs",
#        "//mediapipe/objc:mediapipe_layer_renderer", # no need for layer renderer since I don't render
    ] + select({
        "//conditions:default": [
            "//mediapipe/graphs/face_mesh:mobile_calculators",
            "//mediapipe/framework/formats:landmark_cc_proto",
        ],
    }),
)