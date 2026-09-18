# Copyright 2026 The MediaPipe Authors.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

licenses(["notice"])

SEMANTIC_RETRIEVER_DEPS = [
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/log:absl_check",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:status_macros",
    "@com_google_absl//absl/status:statusor",
    "@com_google_absl//absl/strings",
    "//mediapipe/tasks/c/core:mp_status",
    "//mediapipe/tasks/c/core:mp_status_converter",
    "//mediapipe/tasks/c/core:common",
    "//mediapipe/tasks/c/retrieval/universal_embedder:universal_embedder_lib",
    "//mediapipe/tasks/c/text/text_embedder:text_embedder_c_lib",
    "//mediapipe/tasks/c/vision/core:image",
    "//mediapipe/tasks/c/vision/image_embedder:image_embedder_c_lib",
    "//mediapipe/tasks/cc/text/text_embedder",
    "//mediapipe/tasks/cc/vision/image_embedder",
    "//mediapipe/tasks/cc/core:embedding_provider",
    "//mediapipe/tasks/cc/retrieval/semantic_retriever",
    "//mediapipe/tasks/cc/retrieval/semantic_retriever:memory_store",
    "//mediapipe/tasks/cc/retrieval/semantic_retriever:sqlite_memory_store",
    "//mediapipe/tasks/cc/retrieval/semantic_retriever:text_chunker",
    "//mediapipe/tasks/cc/retrieval/universal_embedder",
]

cc_library(
    name = "semantic_retriever_lib",
    srcs = ["semantic_retriever.cc"],
    hdrs = ["semantic_retriever.h"],
    visibility = ["//visibility:public"],
    deps = SEMANTIC_RETRIEVER_DEPS,
    alwayslink = True,
)
