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
    name = "MPPTaskPart",
    hdrs = [
        "//mediapipe/tasks/ios/core:sources/MPPTaskPart.h",
    ],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskPart",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPEmbedding",
    hdrs = ["//mediapipe/tasks/ios/components/containers:sources/MPPEmbedding.h"],
    deps = [
        "//mediapipe/tasks/ios/components/containers:MPPEmbedding",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPEmbeddingResult",
    hdrs = ["//mediapipe/tasks/ios/components/containers:sources/MPPEmbeddingResult.h"],
    deps = [
        ":MPPEmbedding",
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPTextChunker",
    hdrs = ["sources/MPPTextChunker.h"],
    deps = [
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPDefaultTextChunker",
    srcs = ["sources/MPPDefaultTextChunker.mm"],
    hdrs = ["sources/MPPDefaultTextChunker.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        ":MPPTextChunker",
        "//mediapipe/tasks/cc/text/utils:genai_utils",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPVectorStore",
    hdrs = ["sources/MPPVectorStore.h"],
    deps = [
        ":MPPRetrievalRecord",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPRetrievalRecord",
    srcs = [
        "sources/MPPRetrievalRecord.m",
        "sources/MPPRetrievalResult.m",
    ],
    hdrs = [
        "sources/MPPRetrievalRecord.h",
        "sources/MPPRetrievalResult.h",
    ],
    deps = [
        "//mediapipe/tasks/ios/core:MPPTaskPart",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPSqliteVectorStore",
    srcs = ["sources/MPPSqliteVectorStore.mm"],
    hdrs = ["sources/MPPSqliteVectorStore.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    sdk_dylibs = ["libsqlite3"],
    deps = [
        ":MPPRetrievalRecord",
        ":MPPVectorStore",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever:sqlite_memory_store",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever:sqlite_vector_store",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:memory_cc_proto",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:vector_stores_cc_proto",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:Foundation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

objc_library(
    name = "MPPAudioLoader",
    srcs = ["sources/MPPAudioLoader.mm"],
    hdrs = ["sources/MPPAudioLoader.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    deps = [
        "//mediapipe/tasks/cc/retrieval/semantic_retriever:audio_decoder",
        "//mediapipe/tasks/ios/audio/core:MPPAudioData",
        "//mediapipe/tasks/ios/audio/core:MPPAudioDataFormat",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Foundation",
        "@com_google_absl//absl/strings:string_view",
    ],
)

objc_library(
    name = "MPPSemanticRetrieverComponents",
    srcs = ["sources/MPPSemanticRetrieverComponents.m"],
    hdrs = ["sources/MPPSemanticRetrieverComponents.h"],
    deps = [
        ":MPPDefaultTextChunker",
        ":MPPTextChunker",
        ":MPPVectorStore",
        "//mediapipe/tasks/ios/core:MPPTaskPart",
        "//third_party/apple_frameworks:Foundation",
    ],
)

objc_library(
    name = "MPPSemanticRetriever",
    srcs = ["sources/MPPSemanticRetriever.mm"],
    hdrs = ["sources/MPPSemanticRetriever.h"],
    copts = [
        "-ObjC++",
        "-std=c++20",
        "-Xclang=-fno-cxx-modules",
        "-x objective-c++",
    ],
    module_name = "MPPSemanticRetriever",
    deps = [
        ":MPPAudioLoader",
        ":MPPRetrievalRecord",
        ":MPPSemanticRetrieverComponents",
        ":MPPTaskPart",
        "//mediapipe/framework:timestamp",
        "//mediapipe/tasks/cc/core:tasks_logging_factory",  # build_cleaner: keep
        "//mediapipe/tasks/cc/core/logging:tasks_logger",
        "//mediapipe/tasks/cc/core/logging/factory:logging_factory_hdr",
        "//mediapipe/tasks/cc/text/utils:genai_utils",
        "//mediapipe/tasks/ios/audio/core:MPPAudioData",
        "//mediapipe/tasks/ios/common:MPPCommon",
        "//mediapipe/tasks/ios/common/utils:MPPCommonUtils",
        "//mediapipe/tasks/ios/components/containers:MPPEmbedding",
        "//mediapipe/tasks/ios/components/containers:MPPEmbeddingResult",
        "//mediapipe/tasks/ios/components/utils:MPPCosineSimilarity",
        "//mediapipe/tasks/ios/vision/core:MPPImage",
        "//third_party/apple_frameworks:Foundation",
        "//third_party/apple_frameworks:UIKit",
    ],
)
