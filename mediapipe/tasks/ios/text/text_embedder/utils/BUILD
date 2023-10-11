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
    name = "MPPTextEmbedderOptionsHelpers",
    srcs = ["sources/MPPTextEmbedderOptions+Helpers.mm"],
    hdrs = ["sources/MPPTextEmbedderOptions+Helpers.h"],
    deps = [
        "//mediapipe/framework:calculator_options_cc_proto",
        "//mediapipe/tasks/cc/components/processors/proto:embedder_options_cc_proto",
        "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_cc_proto",
        "//mediapipe/tasks/ios/common/utils:NSStringHelpers",
        "//mediapipe/tasks/ios/core:MPPTaskOptionsProtocol",
        "//mediapipe/tasks/ios/core/utils:MPPBaseOptionsHelpers",
        "//mediapipe/tasks/ios/text/text_embedder:MPPTextEmbedderOptions",
    ],
)

objc_library(
    name = "MPPTextEmbedderResultHelpers",
    srcs = ["sources/MPPTextEmbedderResult+Helpers.mm"],
    hdrs = ["sources/MPPTextEmbedderResult+Helpers.h"],
    deps = [
        "//mediapipe/framework:packet",
        "//mediapipe/tasks/cc/components/containers/proto:embeddings_cc_proto",
        "//mediapipe/tasks/ios/components/containers/utils:MPPEmbeddingResultHelpers",
        "//mediapipe/tasks/ios/text/text_embedder:MPPTextEmbedderResult",
    ],
)
