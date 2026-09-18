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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//mediapipe/tasks:internal"])

licenses(["notice"])

cc_binary(
    name = "libsqlite_vector_store_jni.so",
    linkopts = [
        "-Wl,-soname=libsqlite_vector_store_jni.so",
        "-fvisibility=hidden",
    ],
    linkshared = 1,
    deps = [":retrieval_jni_impl"],
)

cc_library(
    name = "libsqlite_vector_store_jni_lib",
    srcs = [":libsqlite_vector_store_jni.so"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "retrieval_jni_impl",
    srcs = [
        "sqlite_vector_store_jni.cc",
        "text_chunker_jni.cc",
    ],
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever:sqlite_memory_store",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever:sqlite_vector_store",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:vector_stores_cc_proto",
        "//mediapipe/tasks/cc/text/utils:genai_utils",
        "//third_party/sqlite",
        "//util/java:jni_helper",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
    ] + select({
        "//conditions:default": [
            "//third_party/java/jdk:jni",
        ],
        "//mediapipe:android": [],
    }),
    alwayslink = 1,
)
