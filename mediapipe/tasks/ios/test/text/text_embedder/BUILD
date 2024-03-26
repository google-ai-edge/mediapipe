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
    name = "MPPTextEmbedderObjcTestLibrary",
    testonly = 1,
    srcs = ["MPPTextEmbedderTests.m"],
    data = [
        "//mediapipe/tasks/testdata/text:mobilebert_embedding_model",
        "//mediapipe/tasks/testdata/text:regex_embedding_with_metadata",
    ],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/text/text_embedder:MPPTextEmbedder",
    ],
)

ios_unit_test(
    name = "MPPTextEmbedderObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    deps = [
        ":MPPTextEmbedderObjcTestLibrary",
    ],
)

swift_library(
    name = "MPPTextEmbedderSwiftTestLibrary",
    testonly = 1,
    srcs = ["TextEmbedderTests.swift"],
    data = [
        "//mediapipe/tasks/testdata/text:mobilebert_embedding_model",
        "//mediapipe/tasks/testdata/text:regex_embedding_with_metadata",
    ],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/text/text_embedder:MPPTextEmbedder",
    ],
)

ios_unit_test(
    name = "MPPTextEmbedderSwiftTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPTextEmbedderSwiftTestLibrary",
    ],
)
