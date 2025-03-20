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
    name = "MPPAudioEmbedderOptionsHelpers",
    srcs = ["sources/MPPAudioEmbedderOptions+Helpers.mm"],
    hdrs = ["sources/MPPAudioEmbedderOptions+Helpers.h"],
    deps = [
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/tasks/cc/audio/audio_embedder/proto:audio_embedder_graph_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_cc_proto",
        "//mediapipe/tasks/ios/audio/audio_embedder:MPPAudioEmbedderOptions",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskOptionsProtocol",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
    ],
)

objc_library(
    name = "MPPAudioEmbedderResultHelpers",
    srcs = ["sources/MPPAudioEmbedderResult+Helpers.mm"],
    hdrs = ["sources/MPPAudioEmbedderResult+Helpers.h"],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/ios/audio/audio_embedder:MPPAudioEmbedderResult",
        "//mediapipe/tasks/ios/components/containers/utils:MPPEmbeddingResultHelpers",
    ],
)
