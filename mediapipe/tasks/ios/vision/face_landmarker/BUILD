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
    name = "MPPFaceLandmarkerResult",
    srcs = ["sources/MPPFaceLandmarkerResult.mm"],
    hdrs = ["sources/MPPFaceLandmarkerResult.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPFaceLandmarkerOptions",
    srcs = ["sources/MPPFaceLandmarkerOptions.m"],
    hdrs = ["sources/MPPFaceLandmarkerOptions.h"],
    deps = [
        ":MPPFaceLandmarkerResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPFaceLandmarksConnections",
    hdrs = ["sources/MPPFaceLandmarksConnections.h"],
    module_name = "MPPFaceLandmarksConnections",
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPConnection",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPFaceLandmarker",
    srcs = ["sources/MPPFaceLandmarker.mm"],
    hdrs = ["sources/MPPFaceLandmarker.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        ":MPPFaceLandmarkerOptions",
        ":MPPFaceLandmarkerResult",
        ":MPPFaceLandmarksConnections",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPConnection",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/face_landmarker/utils:MPPFaceLandmarkerOptionsHelpers",
        "//mediapipe/tasks/ios/vision/face_landmarker/utils:MPPFaceLandmarkerResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
