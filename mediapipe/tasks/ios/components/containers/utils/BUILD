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

exports_files(glob(["sources/*.h"]))

objc_library(
    name = "MPPCategoryHelpers",
    srcs = ["sources/MPPCategory+Helpers.mm"],
    hdrs = ["sources/MPPCategory+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/ios/components/containers:MPPCategory",
    ],
)

objc_library(
    name = "MPPClassificationResultHelpers",
    srcs = ["sources/MPPClassificationResult+Helpers.mm"],
    hdrs = ["sources/MPPClassificationResult+Helpers.h"],
    deps = [
        ":MPPCategoryHelpers",
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:classification_result",
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
    ],
)

objc_library(
    name = "MPPEmbeddingHelpers",
    srcs = ["sources/MPPEmbedding+Helpers.mm"],
    hdrs = ["sources/MPPEmbedding+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/components/containers:embedding_result",
        "//mediapipe/tasks/ios/components/containers:MPPEmbedding",
    ],
)

objc_library(
    name = "MPPEmbeddingResultHelpers",
    srcs = ["sources/MPPEmbeddingResult+Helpers.mm"],
    hdrs = ["sources/MPPEmbeddingResult+Helpers.h"],
    deps = [
        ":MPPEmbeddingHelpers",
        "//mediapipe/tasks/c/components/containers:embedding_result",
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
    ],
)

objc_library(
    name = "MPPDetectionHelpers",
    srcs = ["sources/MPPDetection+Helpers.mm"],
    hdrs = ["sources/MPPDetection+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/components/containers:category",
        "//mediapipe/tasks/c/components/containers:detection_result",
        "//mediapipe/tasks/c/components/containers:keypoint",
        "//mediapipe/tasks/c/components/containers:rect",
        "//mediapipe/tasks/ios/components/containers:MPPCategory",
        "//mediapipe/tasks/ios/components/containers:MPPDetection",
    ],
)

objc_library(
    name = "MPPLandmarkHelpers",
    srcs = ["sources/MPPLandmark+Helpers.mm"],
    hdrs = ["sources/MPPLandmark+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/components/containers:landmark",
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
    ],
)

objc_library(
    name = "MPPRegionOfInterestHelpers",
    srcs = ["sources/MPPRegionOfInterest+Helpers.mm"],
    hdrs = ["sources/MPPRegionOfInterest+Helpers.h"],
    deps = [
        "//mediapipe/tasks/c/components/containers:keypoint",
        "//mediapipe/tasks/c/vision/interactive_segmenter_legacy:interactive_segmenter_legacy_lib",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/components/containers:MPPDetection",
        "//mediapipe/tasks/ios/components/containers:MPPRegionOfInterest",
    ],
)
