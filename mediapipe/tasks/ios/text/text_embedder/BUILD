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

package(default_visibility = [
    "//mediapipe/tasks:internal",
])

licenses(["notice"])

objc_library(
    name = "MPPTextEmbedderOptions",
    srcs = ["sources/MPPTextEmbedderOptions.m"],
    hdrs = ["sources/MPPTextEmbedderOptions.h"],
    deps = ["//mediapipe/tasks/ios/core:MPPTaskOptions"],
)

objc_library(
    name = "MPPTextEmbedderResult",
    srcs = ["sources/MPPTextEmbedderResult.m"],
    hdrs = ["sources/MPPTextEmbedderResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
    ],
)

objc_library(
    name = "MPPTextEmbedder",
    srcs = ["sources/MPPTextEmbedder.mm"],
    hdrs = ["sources/MPPTextEmbedder.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    module_name = "MPPTextEmbedder",
    deps = [
        ":MPPTextEmbedderOptions",
        ":MPPTextEmbedderResult",
        "//mediapipe/tasks/cc/text/text_embedder:text_embedder_graph",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/components/utils:MPPCosineSimilarity",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//mediapipe/tasks/ios/core:MPPTextPacketCreator",
        "//mediapipe/tasks/ios/text/core:MPPTextTaskRunner",
        "//mediapipe/tasks/ios/text/text_embedder/utils:MPPTextEmbedderOptionsHelpers",
        "//mediapipe/tasks/ios/text/text_embedder/utils:MPPTextEmbedderResultHelpers",
    ],
)
