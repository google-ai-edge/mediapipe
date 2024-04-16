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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

objc_library(
    name = "MPPAudioData",
    srcs = ["sources/MPPAudioData.m"],
    hdrs = ["sources/MPPAudioData.h"],
    deps = [
        ":MPPFloatBuffer",
        ":MPPFloatRingBuffer",
    ],
)

objc_library(
    name = "MPPFloatBuffer",
    srcs = ["sources/MPPFloatBuffer.m"],
    hdrs = ["sources/MPPFloatBuffer.h"],
)

objc_library(
    name = "MPPFloatRingBuffer",
    srcs = ["sources/MPPFloatRingBuffer.m"],
    hdrs = ["sources/MPPFloatRingBuffer.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    deps = [
        ":MPPFloatBuffer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
    ],
)

objc_library(
    name = "MPPAudioRecord",
    srcs = ["sources/MPPAudioRecord.m"],
    hdrs = ["sources/MPPAudioRecord.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    sdk_frameworks = [
        "AVFoundation",
    ],
    deps = [
        ":MPPAudioData",
        ":MPPFloatBuffer",
        ":MPPFloatRingBuffer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
    ],
)
