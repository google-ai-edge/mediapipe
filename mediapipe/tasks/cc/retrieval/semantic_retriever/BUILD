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
