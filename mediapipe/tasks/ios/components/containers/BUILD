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

package(default_visibility = [
    "//mediapipe/tasks:internal",
    "//mediapipe/tasks:users",
])

licenses(["notice"])

objc_library(
    name = "MPPCategory",
    srcs = ["sources/MPPCategory.m"],
    hdrs = ["sources/MPPCategory.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPClassificationResult",
    srcs = ["sources/MPPClassificationResult.m"],
    hdrs = ["sources/MPPClassificationResult.h"],
    deps = [
        ":MPPCategory",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPEmbedding",
    srcs = ["sources/MPPEmbedding.m"],
    hdrs = ["sources/MPPEmbedding.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPEmbeddingResult",
    srcs = ["sources/MPPEmbeddingResult.m"],
    hdrs = ["sources/MPPEmbeddingResult.h"],
    deps = [
        ":MPPEmbedding",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPDetection",
    srcs = ["sources/MPPDetection.m"],
    hdrs = ["sources/MPPDetection.h"],
    deps = [
        ":MPPCategory",
        "//third_party/apple_frameworks:Foundation",
        "//third_party/apple_frameworks:UIKit",
    ],
)

objc_library(
    name = "MPPLandmark",
    srcs = ["sources/MPPLandmark.m"],
    hdrs = ["sources/MPPLandmark.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPConnection",
    srcs = ["sources/MPPConnection.m"],
    hdrs = ["sources/MPPConnection.h"],
    deps = ["//third_party/apple_frameworks:Foundation"],
)

objc_library(
    name = "MPPRegionOfInterest",
    srcs = ["sources/MPPRegionOfInterest.m"],
    hdrs = ["sources/MPPRegionOfInterest.h"],
    deps = [
        ":MPPDetection",
        "//third_party/apple_frameworks:Foundation",
    ],
)
