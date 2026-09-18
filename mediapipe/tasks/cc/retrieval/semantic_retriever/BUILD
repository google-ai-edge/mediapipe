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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "memory_store",
    hdrs = ["memory_store.h"],
    deps = [
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:memory_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_import(
    name = "sqlite_vector_store_prebuilt",
    static_library = select({
        "@platforms//cpu:armv7": "@com_google_mediapipe_android_armeabi_v7a_libsqlite_vector_store_a//file",
        "@platforms//cpu:arm64": "@com_google_mediapipe_android_arm64_v8a_libsqlite_vector_store_a//file",
        "@platforms//cpu:x86_32": "@com_google_mediapipe_android_x86_libsqlite_vector_store_a//file",
        "@platforms//cpu:x86_64": "@com_google_mediapipe_android_x86_64_libsqlite_vector_store_a//file",
        "@bazel_tools//src/conditions:linux_aarch64": "@com_google_mediapipe_desktop_linux_arm64_libsqlite_vector_store_a//file",
        "@bazel_tools//src/conditions:linux_x86_64": "@com_google_mediapipe_desktop_linux_x86_64_libsqlite_vector_store_a//file",
        "@bazel_tools//src/conditions:windows_arm64": "@com_google_mediapipe_desktop_windows_arm64_libsqlite_vector_store_lib//file",
        "@bazel_tools//src/conditions:windows_x64": "@com_google_mediapipe_desktop_windows_x86_64_libsqlite_vector_store_lib//file",
        "@bazel_tools//src/conditions:darwin_arm64": "@com_google_mediapipe_desktop_darwin_arm64_libsqlite_vector_store_a//file",
        "//conditions:default": "@com_google_mediapipe_desktop_linux_x86_64_libsqlite_vector_store_a//file",
    }),
)

cc_library(
    name = "sqlite_vector_store",
    hdrs = ["sqlite_vector_store.h"],
    deps = [
        ":sqlite_vector_store_prebuilt",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:vector_stores_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_import(
    name = "sqlite_memory_store_prebuilt",
    static_library = select({
        "@platforms//cpu:armv7": "@com_google_mediapipe_android_armeabi_v7a_libsqlite_memory_store_a//file",
        "@platforms//cpu:arm64": "@com_google_mediapipe_android_arm64_v8a_libsqlite_memory_store_a//file",
        "@platforms//cpu:x86_32": "@com_google_mediapipe_android_x86_libsqlite_memory_store_a//file",
        "@platforms//cpu:x86_64": "@com_google_mediapipe_android_x86_64_libsqlite_memory_store_a//file",
        "@bazel_tools//src/conditions:linux_aarch64": "@com_google_mediapipe_desktop_linux_arm64_libsqlite_memory_store_a//file",
        "@bazel_tools//src/conditions:linux_x86_64": "@com_google_mediapipe_desktop_linux_x86_64_libsqlite_memory_store_a//file",
        "@bazel_tools//src/conditions:windows_arm64": "@com_google_mediapipe_desktop_windows_arm64_libsqlite_memory_store_lib//file",
        "@bazel_tools//src/conditions:windows_x64": "@com_google_mediapipe_desktop_windows_x86_64_libsqlite_memory_store_lib//file",
        "@bazel_tools//src/conditions:darwin_arm64": "@com_google_mediapipe_desktop_darwin_arm64_libsqlite_memory_store_a//file",
        "//conditions:default": "@com_google_mediapipe_desktop_linux_x86_64_libsqlite_memory_store_a//file",
    }),
)

cc_library(
    name = "sqlite_memory_store",
    hdrs = ["sqlite_memory_store.h"],
    deps = [
        ":memory_store",
        ":sqlite_memory_store_prebuilt",
        ":sqlite_vector_store",
        "//mediapipe/framework/port:status",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:memory_cc_proto",
        "//mediapipe/tasks/cc/retrieval/semantic_retriever/proto:vector_stores_cc_proto",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "text_chunker",
    srcs = ["text_chunker.cc"],
    hdrs = ["text_chunker.h"],
    deps = [
        "//mediapipe/tasks/cc/text/utils:genai_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "text_chunker_test",
    srcs = ["text_chunker_test.cc"],
    deps = [
        ":text_chunker",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status_matchers",
    ],
)

cc_library(
    name = "image_decoder",
    srcs = ["image_decoder.cc"],
    hdrs = ["image_decoder.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "image_decoder_test",
    srcs = ["image_decoder_test.cc"],
    deps = [
        ":image_decoder",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "audio_decoder",
    srcs = ["audio_decoder.cc"],
    hdrs = ["audio_decoder.h"],
    deps = [
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "audio_decoder_test",
    srcs = ["audio_decoder_test.cc"],
    data = [
        "//mediapipe/tasks/testdata/audio:test_audio_clips",
    ],
    deps = [
        ":audio_decoder",
        "//mediapipe/framework/deps:file_path",
        "//mediapipe/framework/port:gtest",
        "//mediapipe/framework/port:gtest_main",
        "//mediapipe/framework/port:status_matchers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "semantic_retriever",
    srcs = ["semantic_retriever.cc"],
    hdrs = ["semantic_retriever.h"],
    deps = [
        ":audio_decoder",
        ":image_decoder",
        ":memory_store",
        ":text_chunker",
        "//mediapipe/framework:timestamp",
        "//mediapipe/tasks/cc/core:embedding_provider",
        "//mediapipe/tasks/cc/core:running_mode",
        "//mediapipe/tasks/cc/core:tasks_logging_factory",  # build_cleaner: keep
        "//mediapipe/tasks/cc/core/logging:tasks_logger",
        "//mediapipe/tasks/cc/core/logging/factory:logging_factory_hdr",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
