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
    name = "MPPPoseLandmarkerResult",
    srcs = ["sources/MPPPoseLandmarkerResult.m"],
    hdrs = ["sources/MPPPoseLandmarkerResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//mediapipe/tasks/ios/vision/core:MPPMask",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPPoseLandmarkerOptions",
    srcs = ["sources/MPPPoseLandmarkerOptions.m"],
    hdrs = ["sources/MPPPoseLandmarkerOptions.h"],
    deps = [
        ":MPPPoseLandmarkerResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPPoseLandmarksConnections",
    hdrs = ["sources/MPPPoseLandmarksConnections.h"],
    module_name = "MPPPoseLandmarksConnections",
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPConnection",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPPoseLandmarker",
    srcs = ["sources/MPPPoseLandmarker.mm"],
    hdrs = ["sources/MPPPoseLandmarker.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    module_name = "MPPPoseLandmarker",
    deps = [
        ":MPPPoseLandmarkerOptions",
        ":MPPPoseLandmarkerResult",
        ":MPPPoseLandmarksConnections",
        "//mediapipe/tasks/cc/vision/pose_landmarker:pose_landmarker_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPConnection",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/pose_landmarker/utils:MPPPoseLandmarkerOptionsHelpers",
        "//mediapipe/tasks/ios/vision/pose_landmarker/utils:MPPPoseLandmarkerResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
