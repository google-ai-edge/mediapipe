# Copyright 2024 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_unit_test",
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
    name = "MPPFloatRingBufferObjcTestLibrary",
    testonly = True,
    srcs = ["MPPFloatRingBufferTests.m"],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        "//mediapipe/tasks/ios/audio/core:MPPFloatRingBuffer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//third_party/apple_frameworks:XCTest",
    ],
)

ios_unit_test(
    name = "MPPFloatRingBufferObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPFloatRingBufferObjcTestLibrary",
    ],
)

objc_library(
    name = "MPPAudioRecordObjcTestLibrary",
    testonly = True,
    srcs = ["MPPAudioRecordTests.mm"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    data = [
        "//mediapipe/tasks/testdata/audio:test_audio_clips",
    ],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        "//mediapipe/tasks/ios/audio/core:MPPAudioRecord",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/test/audio/core/utils:AVAudioPCMBufferTestUtils",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:XCTest",
    ],
)

ios_unit_test(
    name = "MPPAudioRecordObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPAudioRecordObjcTestLibrary",
    ],
)

objc_library(
    name = "MPPAudioDataObjcTestLibrary",
    testonly = True,
    srcs = ["MPPAudioDataTests.mm"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    data = [
        "//mediapipe/tasks/testdata/audio:test_audio_clips",
    ],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        "//mediapipe/tasks/ios/audio/core:MPPAudioData",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/test/audio/core/utils:AVAudioPCMBufferTestUtils",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:XCTest",
    ],
)

ios_unit_test(
    name = "MPPAudioDataObjcTest",
    minimum_os_version = MPP_TASK_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":MPPAudioDataObjcTestLibrary",
    ],
)
