# Copyright 2023 The MediaPipe Authors.
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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPObjectDetectorResult",
    srcs = ["sources/MPPObjectDetectorResult.m"],
    hdrs = ["sources/MPPObjectDetectorResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPDetection",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPObjectDetectorOptions",
    srcs = ["sources/MPPObjectDetectorOptions.m"],
    hdrs = ["sources/MPPObjectDetectorOptions.h"],
    deps = [
        ":MPPObjectDetectorResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPObjectDetector",
    srcs = ["sources/MPPObjectDetector.mm"],
    hdrs = ["sources/MPPObjectDetector.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        ":MPPObjectDetectorOptions",
        ":MPPObjectDetectorResult",
        "//mediapipe/tasks/cc/vision/object_detector:object_detector_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/object_detector/utils:MPPObjectDetectorOptionsHelpers",
        "//mediapipe/tasks/ios/vision/object_detector/utils:MPPObjectDetectorResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
