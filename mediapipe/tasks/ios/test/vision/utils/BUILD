package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPImageTestUtils",
    srcs = ["sources/MPPImage+TestUtils.m"],
    hdrs = ["sources/MPPImage+TestUtils.h"],
    module_name = "MPPImageTestUtils",
    deps = [
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo"],
)

objc_library(
    name = "MPPMaskTestUtils",
    srcs = ["sources/MPPMask+TestUtils.m"],
    hdrs = ["sources/MPPMask+TestUtils.h"],
    module_name = "MPPMaskTestUtils",
    deps = [
        "//mediapipe/tasks/ios/vision/core:MPPMask",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//third_party/apple_frameworks:UIKit",
    ]
)

cc_library(
    name = "parse_proto_utils",
    srcs = ["sources/parse_proto_utils.cc"],
    hdrs = ["sources/parse_proto_utils.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf",
    ],
)
