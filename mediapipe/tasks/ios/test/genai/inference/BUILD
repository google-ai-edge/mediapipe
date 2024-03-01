load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)
load(
    "//mediapipe/framework/tool:ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
)
load(
    "@org_tensorflow//tensorflow/lite:special_rules.bzl",
    "tflite_ios_lab_runner",
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

swift_library(
    name = "LlmInferenceSwiftTestLibrary",
    testonly = 1,
    srcs = ["LlmInferenceTests.swift"],
    data = [
        "//mediapipe/tasks/testdata/genai:test_models",
    ],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        "//mediapipe/tasks/ios/genai/inference:LlmInference",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo"
    ],
)

ios_unit_test(
    name = "LlmInferenceSwiftTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":LlmInferenceSwiftTestLibrary",
    ],
)
