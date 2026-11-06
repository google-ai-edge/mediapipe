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
    name = "MPPObjectDetectorOptionsHelpers",
    srcs = ["sources/MPPObjectDetectorOptions+Helpers.mm"],
    hdrs = ["sources/MPPObjectDetectorOptions+Helpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/c/vision/object_detector:object_detector_lib",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/vision/object_detector:MPPObjectDetectorOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPObjectDetectorResultHelpers",
    srcs = ["sources/MPPObjectDetectorResult+Helpers.mm"],
    hdrs = ["sources/MPPObjectDetectorResult+Helpers.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/c/components/containers:detection_result",
        "//mediapipe/tasks/ios/components/containers/utils:MPPDetectionHelpers",
        "//mediapipe/tasks/ios/vision/object_detector:MPPObjectDetectorResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)
