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
    name = "MPPFaceLandmarkerOptionsHelpers",
    srcs = ["sources/MPPFaceLandmarkerOptions+Helpers.mm"],
    hdrs = ["sources/MPPFaceLandmarkerOptions+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/vision/face_landmarker:face_landmarker_lib",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/vision/face_landmarker:MPPFaceLandmarkerOptions",
    ],
)

objc_library(
    name = "MPPFaceLandmarkerResultHelpers",
    srcs = ["sources/MPPFaceLandmarkerResult+Helpers.mm"],
    hdrs = ["sources/MPPFaceLandmarkerResult+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/vision/face_landmarker:face_landmarker_result",
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
        "//mediapipe/tasks/ios/components/containers/utils:MPPCategoryHelpers",
        "//mediapipe/tasks/ios/components/containers/utils:MPPLandmarkHelpers",
        "//mediapipe/tasks/ios/vision/face_landmarker:MPPFaceLandmarkerResult",
    ],
)
