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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

objc_library(
    name = "MPPImageEmbedderResult",
    srcs = ["sources/MPPImageEmbedderResult.m"],
    hdrs = ["sources/MPPImageEmbedderResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
    ],
)

objc_library(
    name = "MPPImageEmbedderOptions",
    srcs = ["sources/MPPImageEmbedderOptions.m"],
    hdrs = ["sources/MPPImageEmbedderOptions.h"],
    deps = [
        ":MPPImageEmbedderResult",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/vision/core:MPPRunningMode",
    ],
)

objc_library(
    name = "MPPImageEmbedder",
    srcs = ["sources/MPPImageEmbedder.mm"],
    hdrs = ["sources/MPPImageEmbedder.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    module_name = "MPPImageEmbedder",
    deps = [
        ":MPPImageEmbedderOptions",
        ":MPPImageEmbedderResult",
        "//mediapipe/tasks/cc/vision/image_embedder:image_embedder_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/utils:MPPCosineSimilarity",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//mediapipe/tasks/ios/vision/core:MPPVisionPacketCreator",
        "//mediapipe/tasks/ios/vision/core:MPPVisionTaskRunner",
        "//mediapipe/tasks/ios/vision/image_embedder/utils:MPPImageEmbedderOptionsHelpers",
        "//mediapipe/tasks/ios/vision/image_embedder/utils:MPPImageEmbedderResultHelpers",
    ],
)
