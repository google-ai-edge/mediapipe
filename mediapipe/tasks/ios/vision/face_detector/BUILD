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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPFaceDetectorResult",
    srcs = ["sources/MPPFaceDetectorResult.m"],
    hdrs = ["sources/MPPFaceDetectorResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPDetection",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
    ],
)

objc_library(
    name = "MPPFaceDetectorOptions",
    srcs = ["sources/MPPFaceDetectorOptions.m"],
    hdrs = ["sources/MPPFaceDetectorOptions.h"],
    deps = [
        ":MPPFaceDetectorResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPFaceDetector",
    srcs = ["sources/MPPFaceDetector.mm"],
    hdrs = ["sources/MPPFaceDetector.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    deps = [
        ":MPPFaceDetectorOptions",
        ":MPPFaceDetectorResult",
        "//mediapipe/tasks/cc/vision/face_detector:face_detector_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/face_detector/utils:MPPFaceDetectorOptionsHelpers",
        "//mediapipe/tasks/ios/vision/face_detector/utils:MPPFaceDetectorResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
