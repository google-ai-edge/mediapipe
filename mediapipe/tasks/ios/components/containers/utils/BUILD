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
    name = "MPPCategoryHelpers",
    srcs = ["sources/MPPCategory+Helpers.mm"],
    hdrs = ["sources/MPPCategory+Helpers.h"],
    deps = [
        "//mediapipe/framework/formats:classification_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPCategory",
    ],
)

objc_library(
    name = "MPPClassificationResultHelpers",
    srcs = ["sources/MPPClassificationResult+Helpers.mm"],
    hdrs = ["sources/MPPClassificationResult+Helpers.h"],
    deps = [
        ":MPPCategoryHelpers",
        "//mediapipe/tasks/cc/components/containers/proto:classifications_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPClassificationResult",
    ],
)

objc_library(
    name = "MPPEmbeddingHelpers",
    srcs = ["sources/MPPEmbedding+Helpers.mm"],
    hdrs = ["sources/MPPEmbedding+Helpers.h"],
    deps = [
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPEmbedding",
    ],
)

objc_library(
    name = "MPPEmbeddingResultHelpers",
    srcs = ["sources/MPPEmbeddingResult+Helpers.mm"],
    hdrs = ["sources/MPPEmbeddingResult+Helpers.h"],
    deps = [
        ":MPPEmbeddingHelpers",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
    ],
)

objc_library(
    name = "MPPDetectionHelpers",
    srcs = ["sources/MPPDetection+Helpers.mm"],
    hdrs = ["sources/MPPDetection+Helpers.h"],
    deps = [
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:location_data_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPDetection",
    ],
)

objc_library(
    name = "MPPLandmarkHelpers",
    srcs = ["sources/MPPLandmark+Helpers.mm"],
    hdrs = ["sources/MPPLandmark+Helpers.h"],
    deps = [
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/containers:MPPLandmark",
    ],
)

objc_library(
    name = "MPPRegionOfInterestHelpers",
    srcs = ["sources/MPPRegionOfInterest+Helpers.mm"],
    hdrs = ["sources/MPPRegionOfInterest+Helpers.h"],
    deps = [
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/components/containers:MPPRegionOfInterest",
        "//mediapipe/util:color_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
)
