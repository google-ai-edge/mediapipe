load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_unit_test",
)
load(
    "@org_tensorflow//tensorflow/lite:special_rules.bzl",
    "tflite_ios_lab_runner"
)

load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")


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
    tags = [],
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
    minimum_os_version = "11.0",
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags =[],
    deps = [
        ":MPPTextClassifierObjcTestLibrary",
    ],
)

swift_library(
    name = "MPPTextClassifierSwiftTestLibrary",
    testonly = 1,
    srcs = ["TextClassifierTests.swift"],
    tags = [],
)

ios_unit_test(
    name = "MPPTextClassifierSwiftTest",
    minimum_os_version = "11.0",
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = [],
    deps = [
        ":MPPTextClassifierSwiftTestLibrary",
    ],
)
