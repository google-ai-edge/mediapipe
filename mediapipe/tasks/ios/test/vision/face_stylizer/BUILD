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
    name = "MPPFaceStylizerObjcTestLibrary",
    testonly = 1,
    srcs = ["MPPFaceStylizerTests.mm"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    data = [
        "//mediapipe/tasks/testdata/vision:test_images",
        "//mediapipe/tasks/testdata/vision:test_models",
    ],
    deps = [
        "//mediapipe/tasks/ios/test/vision/utils:MPPImageTestUtils",
        "//mediapipe/tasks/ios/vision/face_stylizer:MPPFaceStylizer",
        "//mediapipe/tasks/ios/vision/face_stylizer:MPPFaceStylizerResult",
    ] + select({
        "//third_party:opencv_ios_sim_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_arm64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//third_party:opencv_ios_x86_64_source_build": ["@ios_opencv_source//:opencv_xcframework"],
        "//conditions:default": ["@ios_opencv//:OpencvFramework"],
    }),
)

ios_unit_test(
    name = "MPPFaceStylizerObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPFaceStylizerObjcTestLibrary",
    ],
)
