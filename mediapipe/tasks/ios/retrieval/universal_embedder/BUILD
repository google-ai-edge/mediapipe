# Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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
])

licenses(["notice"])

exports_files(glob(["sources/*.h"]))

objc_library(
    name = "MPPUniversalEmbedderOptions",
    srcs = ["sources/MPPUniversalEmbedderOptions.m"],
    hdrs = ["sources/MPPUniversalEmbedderOptions.h"],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPUniversalEmbedder",
    srcs = ["sources/MPPUniversalEmbedder.mm"],
    hdrs = ["sources/MPPUniversalEmbedder.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPUniversalEmbedder",
    deps = [
        ":MPPUniversalEmbedderOptions",
        "//mediapipe/tasks/c/retrieval/universal_embedder:universal_embedder_lib",
        "//mediapipe/tasks/ios/audio/core:MPPAudioData",
        "//mediapipe/tasks/ios/audio/core:MPPFloatBuffer",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
        "//mediapipe/tasks/ios/components/containers/utils:MPPEmbeddingResultHelpers",
        "//mediapipe/tasks/ios/components/utils:MPPCosineSimilarity",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/core:MPPTaskPart",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/retrieval/semantic_retriever:MPPAudioLoader",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core/utils:MPPImageUtils",
        "//third_party/apple_frameworks:CoreGraphics",  # build_cleaner: keep
        "//third_party/apple_frameworks:CoreMedia",
        "//third_party/apple_frameworks:CoreVideo",
        "//third_party/apple_frameworks:Foundation",
        "//third_party/apple_frameworks:UIKit",
    ],
)
