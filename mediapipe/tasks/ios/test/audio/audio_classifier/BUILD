load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_unit_test",
)
load(
    "@org_tensorflow//tensorflow/lite:special_rules.bzl",
    "tflite_ios_lab_runner",
)
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
)

package(default_visibility = ["//visibility:public"])

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
    name = "MPPAudioClassifierObjcTestLibrary",
    testonly = 1,
    srcs = ["MPPAudioClassifierTests.mm"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    data = [
        "//mediapipe/tasks/testdata/audio:test_audio_clips",
        "//mediapipe/tasks/testdata/audio:test_models",
    ],
    deps = [
        "//mediapipe/tasks/ios/audio/audio_classifier:MPPAudioClassifier",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/test/audio/core/utils:AVAudioFileTestUtils",
        "//mediapipe/tasks/ios/test/audio/core/utils:AVAudioPCMBufferTestUtils",
        "//mediapipe/tasks/ios/test/audio/core/utils:MPPAudioDataTestUtils",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//third_party/apple_frameworks:XCTest",
    ],
)

ios_unit_test(
    name = "MPPAudioClassifierObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPAudioClassifierObjcTestLibrary",
    ],
)
