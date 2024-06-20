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
    name = "MPPHandLandmarkerResult",
    srcs = ["sources/MPPHandLandmarkerResult.m"],
    hdrs = ["sources/MPPHandLandmarkerResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPCategory",
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPHandLandmarkerOptions",
    srcs = ["sources/MPPHandLandmarkerOptions.m"],
    hdrs = ["sources/MPPHandLandmarkerOptions.h"],
    deps = [
        ":MPPHandLandmarkerResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPHandLandmarksConnections",
    hdrs = ["sources/MPPHandLandmarksConnections.h"],
    module_name = "MPPHandLandmarksConnections",
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPConnection",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPHandLandmarker",
    srcs = ["sources/MPPHandLandmarker.mm"],
    hdrs = ["sources/MPPHandLandmarker.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    module_name = "MPPHandLandmarker",
    deps = [
        ":MPPHandLandmarkerOptions",
        ":MPPHandLandmarkerResult",
        ":MPPHandLandmarksConnections",
        "//mediapipe/tasks/cc/vision/hand_landmarker:hand_landmarker_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPConnection",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/hand_landmarker/utils:MPPHandLandmarkerOptionsHelpers",
        "//mediapipe/tasks/ios/vision/hand_landmarker/utils:MPPHandLandmarkerResultHelpers",
        "//third_party/apple_frameworks:Foundation",
    ],
)
