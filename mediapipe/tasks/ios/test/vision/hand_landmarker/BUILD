load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load(
    "@org_tensorflow//tensorflow/lite:special_rules.bzl",
    "tflite_ios_lab_runner",
)
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
)

licenses(["notice"])

# Default tags for filtering iOS targets. Targets are restricted to Apple platforms.
TFL_DEFAULT_TAGS = [
    "apple",
]

# Following sanitizer tests are not supported by iOS test targets.
TFL_DISABLED_SANITIZER_TAGS = [
    "noasan",
    "nomsan",
    "notsan",
]

objc_library(
    name = "MPPHandLandmarkerObjcTestLibrary",
    testonly = True,
    srcs = ["MPPHandLandmarkerTests.m"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    data = [
        "//mediapipe/tasks/testdata/vision:hand_landmarker.task",
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_protos",
    ],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/test/vision/hand_landmarker/utils:MPPHandLandmarkerResultProtobufHelpers",
        "//mediapipe/tasks/ios/test/vision/utils:MPPImageTestUtils",
        "//mediapipe/tasks/ios/vision/hand_landmarker:MPPHandLandmarker",
        "//third_party/apple_frameworks:XCTest",
    ] + select({
        "//third_party:opencv_ios_sim_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_x86_64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//conditions:default": ["@ios_opencv//:OpencvFramework"],
    }),
)

ios_unit_test(
    name = "MPPHandLandmarkerObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPHandLandmarkerObjcTestLibrary",
    ],
)
