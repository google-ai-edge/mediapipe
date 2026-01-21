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
    name = "MPPAudioData",
    srcs = ["sources/MPPAudioData.m"],
    hdrs = ["sources/MPPAudioData.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
    ],
    deps = [
        ":MPPAudioDataFormat",
        ":MPPAudioRecord",
        ":MPPFloatBuffer",
        ":MPPFloatRingBuffer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPAudioDataFormat",
    srcs = ["sources/MPPAudioDataFormat.m"],
    hdrs = ["sources/MPPAudioDataFormat.h"],
    deps = [
        "//third_party/apple_frameworks:AVFAudio",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPFloatBuffer",
    srcs = ["sources/MPPFloatBuffer.m"],
    hdrs = ["sources/MPPFloatBuffer.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPFloatRingBuffer",
    srcs = ["sources/MPPFloatRingBuffer.m"],
    hdrs = ["sources/MPPFloatRingBuffer.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
    ],
    deps = [
        ":MPPFloatBuffer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPAudioRecord",
    srcs = ["sources/MPPAudioRecord.m"],
    hdrs = ["sources/MPPAudioRecord.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
    ],
    deps = [
        ":MPPAudioDataFormat",
        ":MPPFloatBuffer",
        ":MPPFloatRingBuffer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPAudioPacketCreator",
    srcs = ["sources/MPPAudioPacketCreator.mm"],
    hdrs = ["sources/MPPAudioPacketCreator.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
    ],
    deps = [
        ":MPPAudioData",
        "//mediapipe/framework:packet",
        "//mediapipe/framework:timestamp",
        "//mediapipe/framework/formats:matrix",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPAudioRunningMode",
    hdrs = ["sources/MPPAudioRunningMode.h"],
)

objc_library(
    name = "MPPAudioTaskRunner",
    srcs = ["sources/MPPAudioTaskRunner.mm"],
    hdrs = ["sources/MPPAudioTaskRunner.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
    ],
    deps = [
        ":MPPAudioData",
        ":MPPAudioPacketCreator",
        ":MPPAudioRecord",
        ":MPPAudioRunningMode",
        "//mediapipe/framework:packet",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPPacketCreator",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/core:MPPTaskRunner",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Foundation",
    ],
)
