load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")

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

objc_library(
    name = "MPPLlmInferenceObjcTestLibrary",
    testonly = 1,
    srcs = ["MPPLlmInferenceTests.m"],
    data = [
        "//mediapipe/tasks/testdata/genai:test_models",
    ],
    tags = TFL_DEFAULT_TAGS,
)

ios_unit_test(
    name = "MPPLlmInferenceObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPLlmInferenceObjcTestLibrary",
    ],
)
