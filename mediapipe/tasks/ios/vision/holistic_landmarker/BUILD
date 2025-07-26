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

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

objc_library(
    name = "MPPHolisticLandmarkerResult",
    srcs = ["sources/MPPHolisticLandmarkerResult.m"],
    hdrs = ["sources/MPPHolisticLandmarkerResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//mediapipe/tasks/ios/vision/core:MPPMask",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPHolisticLandmarkerOptions",
    srcs = ["sources/MPPHolisticLandmarkerOptions.m"],
    hdrs = ["sources/MPPHolisticLandmarkerOptions.h"],
    deps = [
        ":MPPHolisticLandmarkerResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
    ],
)

objc_library(
    name = "MPPHolisticLandmarker",
    srcs = ["sources/MPPHolisticLandmarker.mm"],
    hdrs = ["sources/MPPHolisticLandmarker.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPHolisticLandmarker",
    deps = [
        ":MPPHolisticLandmarkerOptions",
        ":MPPHolisticLandmarkerResult",
        "//mediapipe/tasks/cc/vision/holistic_landmarker:holistic_landmarker_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/holistic_landmarker/utils:MPPHolisticLandmarkerOptionsHelpers",
        "//mediapipe/tasks/ios/vision/holistic_landmarker/utils:MPPHolisticLandmarkerResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
