load(
    "//mediapipe/tasks:ios/ios.bzl",
    "MPP_TASK_MINIMUM_OS_VERSION",
    "MPP_TASK_DEFAULT_TAGS",
    "MPP_TASK_DISABLED_SANITIZER_TAGS",
)
load(
    "@build_bazel_rules_apple//apple:ios.bzl", 
    "ios_unit_test",
)
load(
    "@org_tensorflow//tensorflow/lite:special_rules.bzl", 
    "tflite_ios_lab_runner"
)

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPTextClassifierObjcTestLibrary",
    testonly = 1,
    srcs = ["MPPTextClassifierTests.m"],
    data = [
        "//mediapipe/tasks/testdata/text:bert_text_classifier_models",
        "//mediapipe/tasks/testdata/text:text_classifier_models",
    ],
    tags = MPP_TASK_DEFAULT_TAGS,
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        "//mediapipe/tasks/ios/text/text_classifier:MPPTextClassifier",
    ],
    
)

ios_unit_test(
    name = "MPPTextClassifierObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = MPP_TASK_DEFAULT_TAGS + MPP_TASK_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPTextClassifierObjcTestLibrary",
    ],
)
