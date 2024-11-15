load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_unit_test",
)
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
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
    name = "MPPTextClassifierObjcTestLibrary",
    testonly = True,
    srcs = ["MPPTextClassifierTests.m"],
    data = [
        "//mediapipe/tasks/testdata/text:bert_text_classifier_models",
        "//mediapipe/tasks/testdata/text:text_classifier_models",
    ],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifier",
        "//third_party/apple_frameworks:XCTest",
    ],
)

ios_unit_test(
    name = "MPPTextClassifierObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPTextClassifierObjcTestLibrary",
    ],
)

swift_library(
    name = "MPPTextClassifierSwiftTestLibrary",
    testonly = True,
    srcs = ["TextClassifierTests.swift"],
    data = [
        "//mediapipe/tasks/testdata/text:bert_text_classifier_models",
        "//mediapipe/tasks/testdata/text:text_classifier_models",
    ],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifier",
        "//third_party/apple_frameworks:XCTest",
    ],
)

ios_unit_test(
    name = "MPPTextClassifierSwiftTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPTextClassifierSwiftTestLibrary",
    ],
)
