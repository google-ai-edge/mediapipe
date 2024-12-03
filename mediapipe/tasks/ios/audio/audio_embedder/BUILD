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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

objc_library(
    name = "MPPAudioEmbedderResult",
    srcs = ["sources/MPPAudioEmbedderResult.m"],
    hdrs = ["sources/MPPAudioEmbedderResult.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
        "//mediapipe/tasks/ios/core:MPPTaskResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPAudioEmbedderOptions",
    srcs = ["sources/MPPAudioEmbedderOptions.m"],
    hdrs = ["sources/MPPAudioEmbedderOptions.h"],
    deps = [
        ":MPPAudioEmbedderResult",
        "//mediapipe/tasks/ios/audio/core:MPPAudioRunningMode",
        "//mediapipe/tasks/ios/core:MPPTaskOptions",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPAudioEmbedder",
    srcs = ["sources/MPPAudioEmbedder.mm"],
    hdrs = ["sources/MPPAudioEmbedder.h"],
    copts = [
        "-ObjC++",
        "-std=c++17",
        "-x objective-c++",
    ],
    deps = [
        ":MPPAudioEmbedderOptions",
        ":MPPAudioEmbedderResult",
        "//mediapipe/tasks/cc/audio/audio_embedder:audio_embedder_graph",
        "//mediapipe/tasks/ios/audio/audio_embedder/utils:MPPAudioEmbedderOptionsHelpers",
        "//mediapipe/tasks/ios/audio/audio_embedder/utils:MPPAudioEmbedderResultHelpers",
        "//mediapipe/tasks/ios/audio/core:MPPAudioData",
        "//mediapipe/tasks/ios/audio/core:MPPAudioRecord",
        "//mediapipe/tasks/ios/audio/core:MPPAudioTaskRunner",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskInfo",
        "//third_party/apple_frameworks:Foundation",
    ],
)
