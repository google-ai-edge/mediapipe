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

load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

objc_library(
    name = "AVAudioPCMBufferTestUtils",
    srcs = ["sources/AVAudioPCMBuffer+TestUtils.m"],
    hdrs = ["sources/AVAudioPCMBuffer+TestUtils.h"],
    deps = [
        "//mediapipe/tasks/ios/audio/core:MPPAudioDataFormat",
        "//mediapipe/tasks/ios/audio/core:MPPFloatBuffer",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "AVAudioFileTestUtils",
    srcs = ["sources/AVAudioFile+TestUtils.m"],
    hdrs = ["sources/AVAudioFile+TestUtils.h"],
    deps = [
        ":AVAudioPCMBufferTestUtils",
        "//mediapipe/tasks/ios/audio/core:MPPAudioData",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPAudioDataTestUtils",
    srcs = ["sources/MPPAudioData+TestUtils.m"],
    hdrs = ["sources/MPPAudioData+TestUtils.h"],
    deps = [
        ":AVAudioPCMBufferTestUtils",
        "//mediapipe/tasks/ios/audio/core:MPPAudioData",
        "//mediapipe/tasks/ios/test/utils:MPPFileInfo",
    ],
)
