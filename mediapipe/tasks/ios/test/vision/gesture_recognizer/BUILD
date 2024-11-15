load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load(
    "@org_tensorflow//tensorflow/lite:special_rules.bzl",
    "tflite_ios_lab_runner",
)
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
)

package(default_visibility = ["//mediapipe/tasks:internal"])

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
    name = "MPPGestureRecognizerObjcTestLibrary",
    testonly = True,
    srcs = ["MPPGestureRecognizerTests.m"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    data = [
        "//mediapipe/tasks/testdata/vision:gesture_recognizer.task",
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_protos",
    ],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/test/vision/gesture_recognizer/utils:MPPGestureRecognizerResultProtobufHelpers",
        "//mediapipe/tasks/ios/test/vision/utils:MPPImageTestUtils",
        "//mediapipe/tasks/ios/vision/gesture_recognizer:MPPGestureRecognizer",
        "//third_party/apple_frameworks:XCTest",
    ] + select({
        "//third_party:opencv_ios_sim_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_x86_64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//conditions:default": ["@ios_opencv//:OpencvFramework"],
    }),
)

ios_unit_test(
    name = "MPPGestureRecognizerObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPGestureRecognizerObjcTestLibrary",
    ],
)
